"""Compaction preserves text, temporal groups, DeepStack alignment and real KV lengths."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen-vl-finetune'))
import unittest
import torch
from evaluation.protocols.token_reduction_runtime import bounded_grid, compact, prepare_features, decoder_greedy, BudgetProcessor

class ReductionTests(unittest.TestCase):
    def fixture(self):
        embeddings=torch.arange(18*8,dtype=torch.float32).reshape(1,18,8)
        mask=torch.tensor([[False]+[True]*8+[False]+[True]*7+[False]])
        positions=torch.arange(18).expand(3,1,18)
        deep=[embeddings[mask].clone(),embeddings[mask].clone()*2]
        return embeddings,positions,mask,deep

    def test_identity_and_text_preservation(self):
        features=self.fixture()
        for method in ('uniform','merge'):
            full=compact(*features,1.,method)
            self.assertIs(full[0],features[0])
            for ratio in (.5,.25,.1):
                e,p,m,d=compact(*features,ratio,method)
                self.assertTrue(torch.equal(e[~m],features[0][~features[2]]))
                self.assertTrue(torch.equal(e[m],d[0]))
                self.assertTrue(torch.equal(d[0]*2,d[1]))
                self.assertEqual(e.shape[1], int(m.sum())+3)
                self.assertGreaterEqual(int(m.sum()),2) # both temporal groups survive

    def test_merge_preserves_every_source_feature(self):
        features=self.fixture()
        e,p,m,d=compact(*features,.1,'merge')
        self.assertTrue(torch.equal(e[0,1],features[0][0,1:9].mean(0)))
        self.assertTrue(torch.equal(e[0,3],features[0][0,10:17].mean(0)))

    def test_pixel_budget(self):
        for h,w in ((1080,1920),(1920,1080),(64,64),(100,900)):
            for count in (576,288,144,58):
                hh,ww=bounded_grid(h,w,count)
                self.assertLessEqual((hh//32)*(ww//32),count)
                self.assertEqual(hh%32+ww%32,0)

class QwenIntegration(unittest.TestCase):
    def model(self):
        from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
        cfg=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=1024,rope_parameters=dict(rope_type='default',rope_theta=10000,mrope_section=[1,1,2])),
            vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,
            patch_size=16,temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=32,
            deepstack_visual_indexes=[0,1]),image_token_id=120,video_token_id=121,
            vision_start_token_id=122,vision_end_token_id=123)
        cfg._attn_implementation='sdpa'
        torch.manual_seed(42)
        return Qwen3VLForConditionalGeneration(cfg).eval()

    def test_full_retention_matches_hf_and_compact_cache(self):
        model=self.model()
        ids=torch.tensor([[1,122,120,120,120,120,123,3,4]])
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.tensor([[0,0,1,1,1,1,0,0,0]]),
            image_grid_thw=torch.tensor([[1,4,4]]),pixel_values=torch.randn(16,3*2*16*16))
        with torch.inference_mode():
            expected=model.generate(**inputs,max_new_tokens=4,do_sample=False,eos_token_id=None,pad_token_id=0)
            features=prepare_features(model,inputs)
            actual,timing=decoder_greedy(model,features,4,set(),fixed=True)
            self.assertTrue(torch.equal(expected[:,ids.shape[1]:],actual))
            for method in ('uniform','merge'):
                reduced=compact(*features,.5,method)
                generated,stats=decoder_greedy(model,reduced,4,set(),fixed=True)
                self.assertEqual(generated.shape,(1,4))
                self.assertEqual(stats['final_kv_tokens'],ids.shape[1]-2+3)

    def test_video_full_retention_and_temporal_groups(self):
        model=self.model()
        ids=torch.tensor([[1,122,121,121,121,121,123,5,122,121,121,121,121,123,4]])
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
            mm_token_type_ids=(ids==121).long()*2,video_grid_thw=torch.tensor([[2,4,4]]),
            pixel_values_videos=torch.randn(32,3*2*16*16))
        with torch.inference_mode():
            expected=model.generate(**inputs,max_new_tokens=4,do_sample=False,eos_token_id=None,pad_token_id=0)
            features=prepare_features(model,inputs)
            actual,_=decoder_greedy(model,features,4,set(),fixed=True)
            self.assertTrue(torch.equal(expected[:,ids.shape[1]:],actual))
            for method in ('uniform','merge'):
                reduced=compact(*features,.25,method)
                self.assertEqual(int(reduced[2].sum()),2)
                generated,stats=decoder_greedy(model,reduced,4,set(),fixed=True)
                self.assertEqual(stats['final_kv_tokens'],ids.shape[1]-6+3)

class PackedWeights(unittest.TestCase):
    def test_integer_code_pack_equivalence(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen-vl-finetune'))
        from qwenvl.quantize.qat_to_awq import pack_qat_linear,verify_pack
        from qwenvl.quantize.awq_linear import dequantize_gemm
        for k,n in ((128,64),(256,160)):
            torch.manual_seed(42)
            scales=torch.full((n*k//128,),-3.)
            codes=torch.randint(-8,8,(n,k))
            weight=codes*scales.exp().reshape(n,-1).repeat_interleave(128,dim=1)
            packed=pack_qat_linear(weight,scales,None,4,128)
            self.assertEqual(verify_pack(packed,weight,scales,4,128),0)
            restored=dequantize_gemm(packed.qweight,packed.qzeros,packed.scales,4,128).T
            torch.testing.assert_close(restored.float(),weight,atol=.001,rtol=.001)
            self.assertEqual(packed.qweight.numel()*8,weight.numel())

    def test_verified_packed_cache_reused_without_repacking(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.protocols.token_reduction_runtime import load_model
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(128,64,bias=False)
                self.config=SimpleNamespace(qat={'num_replaced':1})
            def to(self,*a,**kw):
                return self  # CPU-only test of the packing/cache path.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            scale=torch.full((64,),-3.)
            weight=torch.randint(-8,8,(64,128))*scale.exp()[:,None]
            torch.save({'linear.weight':weight,'linear.log_scale':scale},path/'qat_quantized_weights.bin')
            with patch('transformers.Qwen3VLForConditionalGeneration.from_pretrained',side_effect=lambda *a,**kw:FakeModel()):
                first,audit=load_model(path,True,'sdpa',path/'cache','source-hash')
                self.assertEqual(audit['converted_modules'],1)
                with patch('qwenvl.quantize.qat_to_awq.pack_qat_linear',side_effect=AssertionError('must use cache')):
                    second,audit=load_model(path,True,'sdpa',path/'cache','source-hash')
                self.assertTrue(torch.equal(first.linear.qweight,second.linear.qweight))
                self.assertTrue(torch.equal(first.linear.scales,second.linear.scales))

class ProcessorBudget(unittest.TestCase):
    def test_actual_processor_resolution_budget(self):
        from transformers import AutoProcessor
        from PIL import Image
        root=Path(__file__).resolve().parents[2]
        native=AutoProcessor.from_pretrained(root/'models/Qwen3-VL-2B-Instruct',local_files_only=True)
        image=Image.new('RGB',(1920,1080))
        text=native.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':'Which?'}]}],tokenize=False,add_generation_prompt=True)
        original=None
        for ratio in (1.,.5,.25,.1):
            processor=BudgetProcessor(native,'resolution',ratio)
            inputs=processor(text=[text],images=[image],return_tensors='pt').to('cpu')
            count=int((inputs['input_ids']==native.image_token_id).sum())
            self.assertLessEqual(count,576)
            if ratio==1:
                original=count
            else:
                self.assertEqual(inputs['_baseline_visual_tokens'],original)
                self.assertLessEqual(count,__import__('math').ceil(original*ratio))

if __name__=='__main__':
    unittest.main()
