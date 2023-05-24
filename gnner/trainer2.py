import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
import torch
import pickle
import pandas as pd
from utils import compute_metrics, validate, extract_spans, num_ovelap_span


class LightningWrapper(pl.LightningModule):

    def __init__(self, model,lr):

        super().__init__()

        self.model = model
        self.non_null_labels = list(self.model.map_lab.values())
        self.learning_rate = lr
        self.best_wf1 = -1

        

    def training_step(self, batch, batch_idx):

        x = batch
        batch_len = self.model(x)['batch']
        self.log('Batch', batch_len, prog_bar=True)
        self.log('learning rate', self.learning_rate, prog_bar=False)
        loss = self.model(x)['loss']
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        x = val_batch

        pred = self.model(x)['logits'].argmax(-1)
        
        true = x['span_labels']
        mask = x['span_mask']

        spans = extract_spans(pred, mask, x['original_spans'])

        true, pred = validate(true, pred, mask)

        return true, pred, spans

    def validation_epoch_end(self, val_outs):

        true, pred, spans = [], [], []

        for t, p, s in val_outs:
            true.extend(t)
            pred.extend(p)
            spans.extend(s)
            
        p, r, f, _ = precision_recall_fscore_support(true, pred, labels=self.non_null_labels, average='micro')

        f1_macro = f1_score(true, pred, labels=self.non_null_labels, average='macro')
        
        # Calculate the classification report
        report = classification_report(true, pred, labels=self.non_null_labels, output_dict=True)

        # Print the weighted F1 score
        weighted_f1_score = report['weighted avg']['f1-score']
        
        if weighted_f1_score > self.best_wf1:
            self.best_wf1 = weighted_f1_score
            optimizer = self.configure_optimizers()
            path = "/content/drive/MyDrive/best_model.pth"
            torch.save({
            "model_state_dict" :self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        

        num_ov = num_ovelap_span(spans)
        self.log('W-F1', weighted_f1_score, prog_bar=True)
        self.log('Best W-F1', self.best_wf1, prog_bar=True)
        self.log('macro', f1_macro, prog_bar=True)
        self.log('micro', f, prog_bar=True)
        self.log('overlap', num_ov, prog_bar=True)
        
        self.log('prec', p, prog_bar=True)
        self.log('rec', r, prog_bar=True)
        

    def test_step(self, val_batch, batch_idx):
        return self.validation_step(val_batch, batch_idx)

    def test_epoch_end(self, val_outs):
        
        true, pred, spans = [], [], []

        for t, p, s in val_outs:
            true.extend(t)
            pred.extend(p)
            spans.extend(s)
        
        report = classification_report(true, pred, labels=self.non_null_labels, digits=5)

        num_ov = num_ovelap_span(spans)
        
        with open('log_2.txt', 'a') as f:
            f.write(self.data_path)
            f.write('\n')
            f.write(report)
            f.write(str(num_ov))
            f.write('\n\n')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
