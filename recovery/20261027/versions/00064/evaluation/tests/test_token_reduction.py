"""Compaction preserves text, temporal groups, DeepStack alignment and real KV lengths."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
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

if __name__=='__main__':
    unittest.main()
