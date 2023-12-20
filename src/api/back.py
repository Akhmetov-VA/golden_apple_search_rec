import faiss
import numpy as np
import pandas as pd
import ruclip
import torch
import uvicorn
from fastapi import FastAPI

app = FastAPI()

N_REC = 11

products = pd.read_csv("../../data/products.csv", index_col="sku")
text_embeddings = pd.read_csv("../../data/embeddings/text_ruCLIP_embeddings.csv", index_col="sku")
ru_clip, ru_clip_processor = ruclip.load("ruclip-vit-base-patch32-384", cache_dir="../../ruCLIP_model")
ru_text_index = faiss.read_index("../../data/embeddings/text_ruCLIP_faiss.index")


@app.post("/get_recommendation")
async def get_recommendation(product_index: str):
	_, same_embedding_indexes = ru_text_index.search(
		np.ascontiguousarray(
			text_embeddings
			.loc[str(product_index)]
			.to_numpy()
			.astype("float32")
			.reshape((1, -1))),
		N_REC
	)
	same_sku_indexes = products.iloc[same_embedding_indexes[0][1:11]].index.to_list()

	return {"indexes": same_sku_indexes}


@app.post("/get_search")
async def get_search(user_text_input: str):
	line_tokenized = ru_clip_processor(text=[user_text_input])

	with torch.no_grad():
		line_embedding = ru_clip.encode_text(line_tokenized["input_ids"])

	_, same_embedding_indexes = ru_text_index.search(
		np.ascontiguousarray(
			line_embedding
			.numpy()
		),
		N_REC
	)
	same_sku_indexes = products.iloc[same_embedding_indexes[0][1:11]].index.to_list()

	return {"indexes": same_sku_indexes}


if __name__ == "__main__":
	uvicorn.run(app, host='0.0.0.0', port=8080)
