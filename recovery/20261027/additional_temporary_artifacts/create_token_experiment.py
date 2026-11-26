from pathlib import Path
import json, copy
root=Path('reproduction/GRACE')
old=root/'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/experiment.json'
src=json.loads(old.read_text())
spec={k:copy.deepcopy(src[k]) for k in ('schema','student','bit_widths')}
spec['bit_widths']=[4]
spec['models']={k:copy.deepcopy(src['models'][k]) for k in ('qat_w4','vanilla_2b','released_bf16')}
spec['models']['qat_w4']['path']='experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/w4g128/runs/qat'
spec['models']['qat_w4'].pop('relative_to',None)
spec['models']['qat_w4']['runtime']='strict_awq_int4'
spec['comparisons']=[{'left':'qat_w4','right':'released_bf16','label':'QAT INT4 vs matched BF16 initialization'},{'left':'qat_w4','right':'vanilla_2b','label':'QAT INT4 vs vanilla 2B'}]
spec['report_title']='576-token image/video inference: realized INT4 and visual token reduction'
spec['design']={'retentions':[1.0,0.5,0.25,0.1],'methods':['resolution','uniform','merge'],'image_max_pixels':589824,'image_min_pixels':4096,'video_max_pixels_per_frame':589824,'video_fps':2,'max_new_tokens':256,'seed':42,'video_population':'VideoMME short + medium; TempCompass MC; Video-MMMU three splits','merge_method':'deterministic contiguous spatial bin mean, within each image or temporal patch group','baseline':'one shared 100% job per model/suite; methods identical at retention=1','profile_tokens':32,'profile_samples':3,'profile_warmup':1,'profile_repeats':3}
(root/'experiments/qat_int4_token_reduction_576_seed42/experiment.json').write_text(json.dumps(spec,indent=2)+'\n')
