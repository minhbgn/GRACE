"""Local HTTP tests for resume, deferral, verified cleanup and selective ZIP ranges."""
from pathlib import Path
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
import socket
import struct
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/'data_tools/download_all_data.sh'
source=SCRIPT.read_text().split('# BEGIN EMBEDDED PYTHON\n',1)[1].split('\n# END EMBEDDED PYTHON',1)[0]
spec=importlib.util.spec_from_loader('grace_all_data_downloads',loader=None)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d
exec(compile(source,str(SCRIPT),'exec'),d.__dict__)


class Downloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes={};cls.requests=[];cls.ranges=[];cls.slow=set();cls.refuse=set()
        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):self.serve(True)
            def do_GET(self):self.serve(False)
            def serve(self,head):
                path=self.path
                if path not in cls.routes:self.send_error(404);return
                data=cls.routes[path];start=0;end=len(data)-1;request=self.headers.get('Range')
                cls.requests.append((path,request,head))
                ranged=request and path not in cls.refuse
                if ranged:
                    span=request.split('=')[1].split('-');start=int(span[0]);end=int(span[1]) if span[1] else end
                    if start>len(data)-1:self.send_error(416);return
                self.send_response(206 if ranged else 200)
                self.send_header('Content-Length',str(end-start+1));self.send_header('ETag','fixture')
                self.send_header('Content-Type','application/octet-stream')
                if ranged:
                    self.send_header('Content-Range',f'bytes {start}-{end}/{len(data)}');cls.ranges.append((path,start,end))
                self.end_headers()
                if head:return
                try:
                    for offset in range(start,end+1,4096):
                        if path in cls.slow:time.sleep(.025)
                        self.wfile.write(data[offset:min(offset+4096,end+1)]);self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
            def log_message(self,*args):pass
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.server.daemon_threads=True
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='grace-all-download-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.args=d.parse_args(['--root',str(self.root),'--reserve-gib','0','--grace-seconds','1','--retries','0'])
        self.r=d.Runner(self.args)
        for p in [self.r.state,self.r.parts,self.r.records,self.r.metadata]:p.mkdir(parents=True,exist_ok=True)
        self.r.dataset='fixture';self.r.budget=d.Budget(10_000_000,5,2)
        self.requests.clear();self.ranges.clear();self.routes.clear();self.slow.clear();self.refuse.clear()
    def item(self,path,payload,*,archive=False,layout='keep'):
        self.routes[path]=payload
        return d.Item('fixture:'+path,self.url+path,self.r.parts/Path(path).name if archive else self.root/Path(path).name,
                      len(payload),hashlib.sha256(payload).hexdigest(),archive,self.root/'extracted' if archive else None,layout)
    def zip(self,files,compression=zipfile.ZIP_DEFLATED):
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w',compression=compression) as z:
            for name,data in files.items():z.writestr(name,data)
        return f.getvalue()
    def test_resume_range_and_skip_completed_file(self):
        payload=os.urandom(250_000);item=self.item('/data.bin',payload)
        part=item.path.with_name(item.path.name+'.part');part.write_bytes(payload[:12345])
        d.atomic_json(part.with_name(part.name+'.json'),{'source':item.source,'etag':'fixture'})
        self.r.process(item)
        self.assertEqual(item.path.read_bytes(),payload)
        self.assertIn(('/data.bin','bytes=12345-',False),self.requests)
        count=len(self.requests);self.r.process(item);self.assertEqual(count,len(self.requests))
    def test_delete_zip_only_after_success_and_repair_removed_output(self):
        item=self.item('/images.zip',self.zip({'test/a.jpg':b'one','test/b.jpg':b'two'}),archive=True)
        self.r.process(item);self.assertFalse(item.path.exists());self.assertTrue(self.r.complete(item))
        (item.extract_to/'test/a.jpg').unlink()
        self.r.process(item)
        self.assertEqual((item.extract_to/'test/a.jpg').read_bytes(),b'one');self.assertFalse(item.path.exists())
        self.assertEqual(sum(1 for _,_,h in self.requests if not h),2)
    def test_interrupted_extraction_retains_archive_and_rerun_reuses_it(self):
        item=self.item('/archive.zip',self.zip({'images/a.jpg':b'data'}),archive=True)
        with patch.object(self.r,'extract_entries',side_effect=OSError('disk error')):
            with self.assertRaises(OSError):self.r.process(item)
        self.assertTrue(item.path.exists());self.assertFalse(self.r.complete(item))
        count=len(self.requests);self.r.process(item)
        self.assertEqual(count,len(self.requests));self.assertFalse(item.path.exists())
    def test_archive_traversal_and_tar_symlink_rejected(self):
        item=self.item('/bad.zip',self.zip({'../escape':b'bad'}),archive=True)
        with self.assertRaises(d.Integrity):self.r.process(item)
        self.assertFalse((self.root/'escape').exists());self.assertTrue(item.path.exists())
        f=io.BytesIO()
        with tarfile.open(fileobj=f,mode='w') as t:
            info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='/tmp';t.addfile(info)
        tar=self.item('/bad.tar',f.getvalue(),archive=True)
        with self.assertRaises(d.Integrity):self.r.process(tar)
        self.assertTrue(tar.path.exists())
    def test_no_append_when_server_ignores_range(self):
        payload=b'abcde'*100;item=self.item('/no-range',payload);self.refuse.add('/no-range')
        part=item.path.with_name(item.path.name+'.part');part.write_bytes(payload[:10])
        d.atomic_json(part.with_name(part.name+'.json'),{'source':item.source,'etag':'fixture'})
        with self.assertRaisesRegex(d.Deferred,'byte ranges'):self.r.process(item)
        self.assertEqual(part.read_bytes(),payload[:10]);self.assertFalse(item.path.exists())
    def test_remote_zip_downloads_selected_members_without_long_payload(self):
        payload=self.zip({'data/short.mp4':os.urandom(100_000),'data/long.mp4':os.urandom(200_000),'data/medium.mp4':os.urandom(90_000)})
        archive=self.item('/videos.zip',payload,archive=True)
        remote=d.RemoteZIP(self.r.http,archive.url,len(payload),self.r.budget)
        with zipfile.ZipFile(remote) as z:infos={i.filename:i for i in z.infolist()}
        long=infos['data/long.mp4'];head=payload[long.header_offset:long.header_offset+30]
        fields=struct.unpack('<4s5H3L2H',head);a=long.header_offset+30+fields[-2]+fields[-1];b=a+long.compress_size-1
        self.ranges.clear()
        for name in ['data/short.mp4','data/medium.mp4']:
            info=infos[name];record={k:getattr(info,k) for k in ['filename','header_offset','compress_size','file_size','CRC','compress_type','flag_bits']}
            item=d.Item(archive.source+'#'+name,archive.url,self.root/'videos'/Path(name).name,info.compress_size)
            self.r.zip_member(item,archive,record)
            self.assertTrue(item.path.is_file());self.assertFalse((self.r.parts/(d.key(item.source)+'.compressed.part')).exists())
        self.assertFalse((self.root/'videos/long.mp4').exists())
        self.assertTrue(self.ranges)
        self.assertTrue(all(end<a or start>b for _,start,end in self.ranges))
        self.assertTrue(all(request is not None for _,request,head in self.requests if not head))
    def test_compressed_member_resumes_and_crc_failure_is_not_complete(self):
        data=self.zip({'data/video.mp4':os.urandom(120_000)})
        archive=self.item('/member.zip',data,archive=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:info=z.infolist()[0]
        fields=struct.unpack('<4s5H3L2H',data[info.header_offset:info.header_offset+30]);start=info.header_offset+30+fields[-2]+fields[-1]
        item=d.Item(archive.source+'#'+info.filename,archive.url,self.root/'video.mp4',info.compress_size)
        part=self.r.parts/(d.key(item.source)+'.compressed.part');part.write_bytes(data[start:start+5000])
        record={k:getattr(info,k) for k in ['filename','header_offset','compress_size','file_size','CRC','compress_type','flag_bits']}
        self.r.zip_member(item,archive,record)
        self.assertIn(('/member.zip',f'bytes={start+5000}-{start+info.compress_size-1}',False),self.requests)
        item.path.unlink();record['CRC']^=1
        with self.assertRaises(d.Integrity):self.r.zip_member(item,archive,record)
        self.assertFalse(item.path.exists());self.assertFalse(self.r.complete(item))
    def test_deadline_defers_first_dataset_and_attempts_next(self):
        slow=self.item('/slow',b'x'*2_000_000);fast=self.item('/fast',b'fine');self.slow.add('/slow')
        self.args.grace_seconds=.12;self.args.budget_mbps=1e8
        with patch.object(self.r,'configure_regular',side_effect=lambda n:[slow if n=='slow' else fast]):
            started=time.monotonic();status=self.r.datasets(['slow','fast']);elapsed=time.monotonic()-started
        summary=d.read_json(self.r.state/'latest_summary.json')
        self.assertLess(elapsed,1.0)
        self.assertEqual(status,2);self.assertEqual(summary['slow']['status'],'deferred');self.assertEqual(summary['fast']['status'],'complete')
        self.assertEqual(fast.path.read_bytes(),b'fine');self.assertTrue(slow.path.with_name(slow.path.name+'.part').exists())
    def test_low_disk_defers_before_transfer_and_next_dataset_completes(self):
        big=self.item('/big',b'a'*1000);small=self.item('/small',b'ok')
        with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=100)),patch.object(self.r,'configure_regular',side_effect=lambda n:[big if n=='big' else small]):
            status=self.r.datasets(['big','small'])
        self.assertEqual(status,2);self.assertFalse(big.path.exists());self.assertEqual(small.path.read_bytes(),b'ok')
        self.assertFalse(any(path=='/big' for path,_,_ in self.requests))
    def test_space_accounts_for_download_and_extraction_on_same_filesystem(self):
        item=self.item('/large.zip',self.zip({'a.bin':os.urandom(4096)}),archive=True)
        with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=int(item.size*2))):
            with self.assertRaises(d.Space):self.r.process(item)
        self.assertFalse(self.requests)
    def test_full_sam_list_is_restricted_to_first_51_shards(self):
        links=self.root/'sam.txt';links.write_text('\n'.join(f'sa_{i:06}.tar\\thttps://example.test/{i}.tar' for i in [0,20,50,51,100]))
        self.args.sam_links=links
        with patch.object(self.r,'direct',side_effect=lambda *a,**kw:(a,kw)):
            items=self.r.sam_items()
        self.assertEqual([i[0][0] for i in items],['sam-shard-000000','sam-shard-000020','sam-shard-000050'])
        self.assertIn(1,self.r.missing_sam);self.assertNotIn(51,self.r.missing_sam)
    def test_completion_is_bound_to_output_destination(self):
        item=self.item('/file',b'original');self.r.process(item)
        other=d.Item(item.source,item.url,self.root/'other/file',item.size,item.sha256)
        self.assertFalse(self.r.complete(other))
        self.r.process(other);self.assertEqual(other.path.read_bytes(),b'original')
    def test_dataset_preflight_includes_all_retained_outputs(self):
        items=[self.item('/first',b'a'*60),self.item('/second',b'b'*60)]
        with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            with self.assertRaises(d.Space):self.r.set_budget(items)
        self.assertFalse(self.requests)
    def test_expired_budget_leaves_existing_signal_handler_intact(self):
        original=d.signal.getsignal(d.signal.SIGALRM)
        with self.assertRaises(d.Deadline):
            with d.Budget(0,5,0).guard():pass
        self.assertIs(d.signal.getsignal(d.signal.SIGALRM),original)
    def test_layouts_match_training_guide(self):
        self.assertEqual(d.Runner.layout_name('00011/a.jpg','llava'),'images/00011/a.jpg')
        self.assertEqual(d.Runner.layout_name('images/00011/a.jpg','llava'),'images/00011/a.jpg')
        self.assertEqual(d.Runner.layout_name('sa_000020/sa_42.jpg','sam'),'images/sa_42.jpg')
        self.assertEqual(d.Runner.layout_name('train_images/a.jpg','textvqa'),'train_images/a.jpg')
        self.assertEqual(d.Runner.layout_name('images/a.jpg','web:wikiart'),'wikiart/images/a.jpg')
        self.assertEqual(d.Runner.layout_name('data/web-landmark/images/a.jpg','web'),'web-landmark/images/a.jpg')
    def test_web_download_selects_only_four_training_archives(self):
        names=['playground.zip','sam_images_share-sft.zip','share_textvqa.zip','web-celebrity.zip','web-landmark.zip','wikiart.zip']
        rows=[{'id':str(i),'path':name} for i,name in enumerate(names)]
        with patch.object(self.r,'drive_listing',return_value=rows):items=self.r.configure_regular('training-web')
        self.assertEqual({i.path.name for i in items},{'web-'+name for name in names[2:]})
        with patch.object(self.r,'drive_listing',return_value=rows[:-1]):
            with self.assertRaisesRegex(d.Deferred,'wikiart.zip'):self.r.configure_regular('training-web')
    def test_default_hour_budget_shared_between_dataset_phases(self):
        self.r.args=d.parse_args(['--root',str(self.root),'--reserve-gib','0'])
        self.r.budget=None
        self.r.set_budget([]);original=self.r.budget
        self.assertEqual(original.seconds,3600)
        original.end-=10
        self.r.set_budget([])
        self.assertIs(self.r.budget,original);self.assertLess(self.r.budget.left(),3591)
        self.r.budget=None;self.r.args.budget_mbps=.0001
        self.r.set_budget([self.item('/large-budget',b'x'*1_000_000)])
        self.assertEqual(self.r.budget.seconds,3600)
    def test_repair_legacy_llava_without_redownload(self):
        base=self.root/'llava';base.mkdir();(base/'00011').mkdir()
        old=base/'00011/a.jpg';old.write_bytes(b'image')
        item=d.Item('llava','',self.root/'images.zip',archive=True,extract_to=base,layout='llava')
        self.r.done(item,[self.r.file_record(old,hashlib.sha256(b'image').hexdigest())])
        self.r.repair_llava_layout(item)
        self.assertTrue((base/'images/00011/a.jpg').is_file());self.assertFalse(old.exists())
        self.assertTrue(self.r.complete(item));self.r.repair_llava_layout(item)
        # Recover a run interrupted after renaming the final shard, before updating its record.
        record=d.read_json(self.r.marker(item));record['outputs'][0]['path']=str(old)
        d.atomic_json(self.r.marker(item),record);self.r.repair_llava_layout(item)
        self.assertTrue(self.r.complete(item));self.assertFalse(self.requests)
    def test_helper_timeout_terminates_child_and_keeps_partial(self):
        part=self.root/'helper.part';code='from pathlib import Path;import sys,time;Path(sys.argv[1]).write_bytes(b"partial");time.sleep(30)'
        with self.assertRaises(d.Deadline):self.r.run_child([sys.executable,'-c',code,str(part)],.2)
        self.assertEqual(part.read_bytes(),b'partial');self.assertIsNone(self.r.current_process)


if __name__=='__main__':unittest.main()
