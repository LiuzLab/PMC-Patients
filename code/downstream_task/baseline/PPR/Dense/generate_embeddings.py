import json
import numpy as np
import torch
import os
import faiss
from tqdm import trange, tqdm
from transformers import AutoTokenizer
from model import BiEncoder
from beir.retrieval.evaluation import EvaluateRetrieval
import argparse


def generate_embeddings(tokenizer, model, patients, device, output_dir=None, model_max_length=512, batch_size=1024, train_data_path=None):
    print(f"batch_size: {batch_size}")
    # Load test patient UIDs from qrels_test.tsv
    train_patient_uids = json.load(open("../../../../../datasets/meta_data/train_patient_uids.json", "r"))
    with open("../../../../../datasets/PPR/qrels_test.tsv", "r") as f:
        lines = f.readlines()
    test_set = set([line.split('\t')[0] for line in lines[1:]])
    test_patient_uids = []
    test_patients = []
    
    # Process test patients
    for patient in test_set:
        test_patient_uids.append(patient)
        test_patients.append(patients[patient]['patient'])
    
    # Load train patient data
    train_patient_uids = []
    train_patients = []
    
    if train_data_path and os.path.exists(train_data_path):
        print(f"Loading train data from {train_data_path}")
        # Parse the JSONL file for training data
        with open(train_data_path, 'r') as file:
            for line_number, line in enumerate(tqdm(file, desc="Processing training data")):
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Parse the JSON object
                    json_obj = json.loads(line)
                    #json_id = json_obj.get("id", f"line_{line_number}")
                    custom_id = json_obj.get("custom_id", f"line_{line_number}")
                    json_id = custom_id.replace("request-", "") if isinstance(custom_id, str) and custom_id.startswith("request-") else custom_id
                    # Extract the response body if it exists
                    if "response" in json_obj and "body" in json_obj["response"]:
                        body = json_obj["response"]["body"]
                        
                        # Check if 'choices' exists and is a list with at least one element
                        if "choices" in body and isinstance(body["choices"], list) and len(body["choices"]) > 0:
                            # Get the assistant message content
                            message_content = body["choices"][0].get("message", {}).get("content", "")
                            
                            if message_content:
                                try:
                                    # Parse the JSON content from the message
                                    parsed_content = json.loads(message_content)
                                    
                                    # Extract chunked_text_rows
                                    if "chunked_text_rows" in parsed_content:
                                        for chunk_idx, chunk_data in enumerate(parsed_content["chunked_text_rows"]):
                                            if "chunk_text" in chunk_data:
                                                # Create a unique ID for this chunk
                                                chunk_id = f"{json_id}_{chunk_idx}"
                                                train_patient_uids.append(json_id)
                                                train_patients.append(chunk_data["chunk_text"])
                                except json.JSONDecodeError:
                                    print(f"Warning: Could not parse JSON from message content in line {line_number}")
                except Exception as e:
                    print(f"Error processing line {line_number}: {str(e)}")
    else:
        # Fallback to original method if no train_data_path is provided
        print("Using default train patient UIDs")
        train_patient_uids = json.load(open("../../../../../meta_data/train_patient_uids.json", "r"))
        train_patients = [patients[patient]['patient'] for patient in train_patient_uids]

    print(f"Loaded {len(test_patients)} test patients and {len(train_patients)} train patients")

    # Generate embeddings
    model.eval()
    with torch.no_grad():
        # Process test embeddings
        print("Generating test embeddings...")
        tokenized = tokenizer(test_patients, max_length=model_max_length, padding="max_length", truncation=True, return_tensors='pt')
        test_embeddings = process_batch(model, tokenized, 0, min(batch_size, len(test_patients)), device, output_dir)
        
        for i in trange(1, (len(test_patients) // batch_size) + 1):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_patients))
            if start_idx >= end_idx:
                break
                
            temp = process_batch(model, tokenized, start_idx, end_idx, device, output_dir)
            test_embeddings = np.concatenate((test_embeddings, temp), axis=0)
        
        print(f"Test embeddings shape: {test_embeddings.shape}")

        # Process train embeddings
        print("Generating train embeddings...")
        tokenized = tokenizer(train_patients, max_length=model_max_length, padding="max_length", truncation=True, return_tensors='pt')
        train_embeddings = process_batch(model, tokenized, 0, min(batch_size, len(train_patients)), device, output_dir)
        
        for i in trange(1, (len(train_patients) // batch_size) + 1):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_patients))
            if start_idx >= end_idx:
                break
                
            temp = process_batch(model, tokenized, start_idx, end_idx, device, output_dir)
            train_embeddings = np.concatenate((train_embeddings, temp), axis=0)
        
        print(f"Train embeddings shape: {train_embeddings.shape}")

    # Save embeddings and UIDs if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "test_embeddings.npy"), test_embeddings)
        np.save(os.path.join(output_dir, "train_embeddings.npy"), train_embeddings)
        json.dump(test_patient_uids, open(os.path.join(output_dir, "test_patient_uids.json"), "w"), indent=4)
        json.dump(train_patient_uids, open(os.path.join(output_dir, "train_patient_uids.json"), "w"), indent=4)

    return test_embeddings, test_patient_uids, train_embeddings, train_patient_uids


# Helper function to process batches
def process_batch(model, tokenized, start_idx, end_idx, device, output_dir):
    temp = model.encoder(
        input_ids=tokenized['input_ids'][start_idx:end_idx].to(device),
        attention_mask=tokenized["attention_mask"][start_idx:end_idx].to(device),
        token_type_ids=tokenized["token_type_ids"][start_idx:end_idx].to(device)
    )
    
    if output_dir:    
        temp = temp[1].detach().cpu().numpy()
    else:
        temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
    
    return temp


def dense_retrieve(queries, query_ids, documents, doc_ids, nlist = 1024, m = 24, nprobe = 100):
    dim = queries.shape[1]

    k = 10000
    quantizer = faiss.IndexFlatIP(dim)
    # Actually for PPR, it is possible to perform exact search.
    index = faiss.IndexFlatIP(dim)
    #index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    #index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)

    print(index.is_trained)
    index.train(documents)
    print(index.is_trained)
    index.add(documents)
    index.nprobe = nprobe
    print(index.ntotal)  

    qrels = {}
    with open("../../../../../datasets/PPR/qrels_test.tsv", "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        q, doc, score = line.split('\t')
        if q in qrels:
            qrels[q][doc] = int(score)
        else:
            qrels[q] = {doc: int(score)}

    print("Begin search...")
    results = index.search(queries, k)
    print("End search!")

    retrieved = {}
    for i in range(results[1].shape[0]):
        result_ids = [doc_ids[idx] for idx in results[1][i]]
        result_scores = results[0][i]
        retrieved[query_ids[i]] = {result_ids[j]: float(result_scores[j]) for j in range(k)}

    #json.dump(retrieved, open("../PPR_link_test.json", "w"), indent = 4)
    evaluation = EvaluateRetrieval()
    metrics = evaluation.evaluate(qrels, retrieved, [10, 1000])
    mrr = evaluation.evaluate_custom(qrels, retrieved, [index.ntotal], metric="mrr")
    return mrr[f'MRR@{index.ntotal}'], metrics[3]['P@10'], metrics[0]['NDCG@10'], metrics[2]['Recall@1000']


def run_metrics(output_dir):
    test_embeddings = np.load("{}/test_embeddings.npy".format(output_dir))
    train_embeddings = np.load("{}/train_embeddings.npy".format(output_dir))
    test_patient_uids = json.load(open("{}/test_patient_uids.json".format(output_dir), "r"))
    train_patient_uids = json.load(open("{}/train_patient_uids.json".format(output_dir), "r"))
    results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
    print(results)
    return


def run_unsupervised(model_name_or_path, output_dir=None, train_data_path=None, have_embeddings=False):    
    # torch.distributed.init_process_group(backend = "nccl", init_method = 'env://')
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    # print(local_rank, device)
    device = "cuda"

    model = BiEncoder(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    # model=torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)
    # model.module.load_state_dict(torch.load("{}/best_model.pth".format(output_dir)))

    patients = json.load(open("../../../../../datasets/PMC-Patients.json", "r"))
    patients = {patient['patient_uid']: patient for patient in patients}

    # If embeddings are not present, generate them
    if not have_embeddings:
        test_embeddings, test_patient_uids, train_embeddings, train_patient_uids = generate_embeddings(
            tokenizer, model, patients, device, output_dir, batch_size=1024, train_data_path=train_data_path
        )
    else:
        test_embeddings = np.load("{}/test_embeddings.npy".format(output_dir))
        train_embeddings = np.load("{}/train_embeddings.npy".format(output_dir))
        test_patient_uids = json.load(open("{}/test_patient_uids.json".format(output_dir), "r"))
        train_patient_uids = json.load(open("{}/train_patient_uids.json".format(output_dir), "r"))
    results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
    print(results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for patient retrieval")
    parser.add_argument("--model", type=str, default="./MedCPT-d", 
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save embeddings")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to JSONL file containing train data")
    parser.add_argument("--have_embeddings", type=bool, default=False,
                        help="Whether to skip generating embeddings")
    
    args = parser.parse_args()
    
    #model_name_or_path = "michiyasunaga/BioLinkBERT-base"
    #model_name_or_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    #model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
    #model_name_or_path = "allenai/specter"
    model_name_or_path = args.model

    #output_dir = "output_linkbert"
    output_dir = args.output_dir
    #run_metrics(output_dir)
    run_unsupervised(model_name_or_path, output_dir, args.train_data, args.have_embeddings)
