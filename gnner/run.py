from models import GnnerAT, GnnerCONV
from trainer import LightningWrapper
import pickle
import pandas as pd
import argparse
import pytorch_lightning as pl
import wandb

def run(model_name, epoch,batch_size,learning_rate):
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
	model = GnnerCONV(labels=labels,model_name=model_name)

	train_loader = model.create_dataloader(train_data, batch_size=batch_size, num_workers=4, shuffle=False)
	val_loader = model.create_dataloader(dev_data, batch_size=batch_size, num_workers=4, shuffle=False)

	lightning = LightningWrapper(model,lr=learning_rate)

	import time
	print('The current local time is :', time.ctime())
	local = ''.join(str(time.ctime()).split())
	run_name = model_name+local

	# logging
	use_wandb = True
	wandb.login(key='0ab9c5b0bd0a9fa4efcc06cff7109f7e9a098bd9')

	if use_wandb:
		from pytorch_lightning.loggers import WandbLogger
		logger = WandbLogger(project="FiNER-GCN",name=run_name)
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
	args = parser.parse_args()

	run(model_name=args.model_name,epoch=args.epoch,batch_size=args.batch,learning_rate=args.lr)

if __name__ == "__main__":
 	main()
