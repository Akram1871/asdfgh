from models2 import GnnerAT, GnnerCONV
from trainer2 import LightningWrapper
import pickle
import pandas as pd
import argparse
import pytorch_lightning as pl
import wandb

def run(model_name, epoch,batch_size,learning_rate, max_span, emb_width, project_dim):
	load_train_df = pd.read_pickle("finer_train_with_val_data3.pkl")
	load_test_df = pd.read_pickle("finer_test_data3.pkl")
	train_data = load_train_df.values
	dev_data = load_test_df.values
	#labels=['C', 'E']
	labels = ['PER','LOC','ORG']
	# import pandas as pd
	# df = pd.DataFrame(dev_data)
	# df.head()


	#encoder='allenai/scibert_scivocab_uncased'
	# model = GnnerAT(labels=labels,model_name=model_name)
	model = GnnerCONV(labels=labels,model_name=model_name,max_span_width=max_span, width_embedding_dim=emb_width, project_dim=project_dim)

	train_loader = model.create_dataloader(train_data, batch_size=batch_size, num_workers=4, shuffle=False)
	val_loader = model.create_dataloader(dev_data, batch_size=batch_size, num_workers=4, shuffle=False)
	

  
	lightning = LightningWrapper(model,lr=learning_rate)
  

	import time
	print('The current local time is :', time.ctime())
	local = ''.join(str(time.ctime()).split())
	run_name = model_name+local+'maxpool'+'Span-'+str(max_span)+"emb_dim-"+str(emb_width)+"project-"+str(project_dim)

	# logging
	use_wandb = True
	wandb.login(key='0ab9c5b0bd0a9fa4efcc06cff7109f7e9a098bd9')

	if use_wandb:
		from pytorch_lightning.loggers import WandbLogger
		logger = WandbLogger(project="FiNER-GCN-GAT-BiLSTM",name=run_name)
	else:
		logger = None


	trainer = pl.Trainer(logger=logger, max_epochs=epoch, gpus=1)

	trainer.fit(lightning, train_loader, val_loader)
	
def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_name", default="bert-base-uncased", type=str,required=True, help="The model name should be from huggingface models")

  parser.add_argument("--epoch",default=40,type=int,required=True,help="The number_of epoch")

  parser.add_argument("--batch",default=16,type=int,required=True,help="The number_of Batch size")
  parser.add_argument("--lr",default=1e-5,type=float,required=True,help="Enter Learning rate")
  parser.add_argument("--max_span",default=8,type=int,required=True,help="The max_span width")  
  parser.add_argument("--emb_width",default=128,type=int,required=True,help="The number_of emb_width")
  parser.add_argument("--project_dim",default=256,type=int,required=True,help="The number_of project_dim")
  args = parser.parse_args()

  run(model_name=args.model_name,epoch=args.epoch,batch_size=args.batch,learning_rate=args.lr,max_span=args.max_span, emb_width=args.emb_width, project_dim=args.project_dim)

if __name__ == "__main__":
 	main()
