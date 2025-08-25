# train_fold_worker.py
# not used. Originally aimed to train my cross validation folds in 
# parallel but that became too cumbersome for the payoff on kaggle.
import os, argparse, json
from datasets import load_from_disk, Audio

def main(gpu, fold_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    from transformers import (Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                              Wav2Vec2Processor, TrainingArguments, Trainer, EarlyStoppingCallback,
                              DataCollatorCTCWithPadding)
    import numpy as np
    try:
        SPANISH = ["EBVS","ERMS","MBMPS","NJS"]
        spkr = SPANISH[fold_idx]
        fold_dir = f"/kaggle/input/l2-arctic-phoneme-data-prep-for-wav2vec2/spanish_loso_es/fold_{spkr}/"

        dset = load_from_disk(fold_dir).cast_column("audio", Audio(sampling_rate=16000))
        train_ds, val_ds, test_ds = dset["train"], dset["validation"], dset["test"]

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "/kaggle/input/l2-arctic-phoneme-data-prep-for-wav2vec2/",
            unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        feat = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                                        padding_value=0.0, do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feat, tokenizer=tokenizer)
        collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        # Fallback CER (avoids internet for evaluate.load)
        def _lev(a,b):
            dp=list(range(len(b)+1))
            for i,ca in enumerate(a,1):
                prev,dp[0]=dp[0],i
                for j,cb in enumerate(b,1):
                    cur=dp[j]
                    dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(ca!=cb)); prev=cur
            return dp[-1]
        def compute_metrics(pred):
            ids = np.argmax(pred.predictions, axis=-1)
            hyp = processor.batch_decode(ids)
            lab = pred.label_ids.copy()
            lab[lab == -100] = processor.tokenizer.pad_token_id
            ref = processor.batch_decode(lab, group_tokens=False)
            h = ["".join(s.split()) for s in hyp]
            r = ["".join(s.split()) for s in ref]
            num = sum(_lev(a,b) for a,b in zip(h,r))
            den = sum(max(1,len(b)) for b in r)
            return {"cer": num/den}

        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            attention_dropout=0.1, layerdrop=0.0, feat_proj_dropout=0.0,
            mask_time_prob=0.75, mask_time_length=10, mask_feature_prob=0.25, mask_feature_length=64,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
        model.freeze_feature_encoder()

        out_dir = f"/kaggle/working/spanish_cv/{spkr}"
        args = TrainingArguments(
            output_dir=out_dir, group_by_length=True,
            per_device_train_batch_size=4, gradient_accumulation_steps=1,
            evaluation_strategy="epoch", save_strategy="epoch",
            gradient_checkpointing=True, fp16=True,
            logging_steps=100, learning_rate=3e-5, warmup_steps=2000,
            save_total_limit=2, num_train_epochs=30,
            load_best_model_at_end=True, metric_for_best_model="cer",
            greater_is_better=False, report_to="none",
            dataloader_num_workers=2, dataloader_persistent_workers=True,
            dataloader_pin_memory=True, seed=42,
        )

        trainer = Trainer(
            model=model, args=args, data_collator=collator,
            compute_metrics=compute_metrics, train_dataset=train_ds,
            eval_dataset=val_ds, tokenizer=processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        )
        trainer.train()
        test_metrics = trainer.predict(test_ds).metrics
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/test_metrics.json","w") as f: json.dump(test_metrics,f,indent=2)
        print("DONE", spkr, test_metrics)
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        with open(f"/kaggle/working/ERR_{fold_idx}.log","w") as f: f.write(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    a = ap.parse_args()
    main(a.gpu, a.fold_idx)
