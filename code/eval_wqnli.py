import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm
import argparse 

def main(args):
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/" + args.model.split("/")[-1].split("-finetuned")[0])

    # sensible default output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.model + "_wqnli_results.csv"
    
    wq_nli = pd.read_csv('./winoqueer_nli.csv')

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    print("loaded model and data")

    # labels
    # 0 = entailment
    # 1 = neutral
    # 2 = contradict

    # output DF 
    # stereo_prem | counter_prem | hyp | p(ent|stereo) | p(neut|stereo) | p(contra|stereo) |
    #p (ent|counter) | p(neut|counter) | p(contra|counter)

    counts_stereo = [0, 0, 0]
    counts_counter = [0, 0, 0]
    count_worse = 0
    output_rows = []
    output_df = pd.DataFrame(columns = ['stereo_premise', 'counter_premise',
                                        'hypothesis', 'p(ent|stereo)',
                                        'p(neutral|stereo)', 'p(contra|stereo)',
                                        'p(ent|counter)', 'p(neut|counter)', 'p(contra|counter)'])
    softmax = torch.nn.Softmax(dim=1)
    for index, row in tqdm(wq_nli.iterrows(), total=wq_nli.shape[0]):
        inputs_stereo = tokenizer(row['stereo_premise'], row['hypothesis'], return_tensors="pt")
        inputs_counter =  tokenizer(row['counter_premise'], row['hypothesis'], return_tensors="pt")
        if torch.cuda.is_available():
            inputs_stereo.to('cuda')
            inputs_counter.to('cuda')
        with torch.no_grad():
            logits_stereo = model(**inputs_stereo).logits
            logits_counter = model(**inputs_counter).logits
            probs_stereo = softmax(logits_stereo)
            probs_counter = softmax(logits_counter)
            predicted_stereo = logits_stereo.argmax().item()
            predicted_counter = logits_counter.argmax().item()

        if probs_stereo[0, 0] > probs_counter[0, 0]:
            count_worse += 1
        counts_stereo[predicted_stereo] += 1
        counts_counter[predicted_counter] += 1
        out_row = [row['stereo_premise'], row['counter_premise'], row['hypothesis'], probs_stereo[0, 0].item(), probs_stereo[0, 1].item(), probs_stereo[0, 2].item(), probs_counter[0, 0].item(), probs_counter[0, 1].item(), probs_counter[0, 2].item()]
        output_rows.append(out_row)

        # write to disk every 10000 rows
        if index != 0 and index % 10000 == 0:
            output_df = pd.concat([output_df, pd.DataFrame(output_rows, columns=output_df.columns)])
            output_df.to_csv(output_file)
            output_rows = []


    # last disk write
    output_df = pd.concat([output_df, pd.DataFrame(output_rows, columns=output_df.columns)]) #added columns parameter
    output_df.to_csv(output_file)

    print(counts_stereo)
    print(count_worse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model location")
    parser.add_argument("--tokenizer", type=str, help="tokenizer location (if different from model location")
    parser.add_argument("--output", type=str, help="output file location")
    args = parser.parse_args()
    main(args)
