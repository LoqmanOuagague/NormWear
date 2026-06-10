import argparse
import os
import gc
import sys
import json
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from .msitf_fusion import *
from ..downstream_pipeline.task_specification import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import  accuracy_score, precision_score, f1_score
from dotenv import load_dotenv

#from .msitf_fusion import _resolve_tinyllama_path
load_dotenv()
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)

DEFAULT_MSITF_CKPT = os.getenv("MSITF_CKPT_PATH")
DEFAULT_MODEL_CKPT = os.getenv("MODEL_CKPT_PATH")

class NormWearZeroShotHF(NormWearZeroShot):
    def __init__(
            self, 
            use_query=True,
            rel_only=False,
            hf_model_id="mosaic-laboratory/normwear"
        ):
        nn.Module.__init__(self)
        self.use_query = use_query
        self.rel_only = rel_only
        tinyllama_path = resolve_tinyllama_path()
        self.tokenizer = AutoTokenizer.from_pretrained(tinyllama_path)
        self.nlp_model = freeze_model(AutoModelForCausalLM.from_pretrained(tinyllama_path))

        self.query_size = 2048

        # compatibility with PyTorch/hf types
        torch.uint64 = torch.int64
        torch.uint32 = torch.int32
        torch.uint16 = torch.int16

        import transformers
        orig_get_init_context = transformers.PreTrainedModel.get_init_context
        transformers.PreTrainedModel.get_init_context = lambda *args, **kwargs: [
            c for c in orig_get_init_context(*args, **kwargs)
            if not (isinstance(c, torch.device) and c.type == 'meta')
        ]

        
        hf_model = transformers.AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)
        
        transformers.PreTrainedModel.get_init_context = orig_get_init_context

        self.sensor_model = freeze_model(hf_model.normwear)
        self.aggregator = freeze_model(hf_model.normwear.msitf_aggregator)
        print(f"NormWear and MSiTF loaded from Hugging Face repo: {hf_model_id}")

        # loss
        loss_l1 = nn.L1Loss()
        loss_cos = nn.CosineEmbeddingLoss()
        self.lambda_temp = nn.Parameter(torch.ones(1)*42, requires_grad=True)
        self.loss_f = lambda x, y: torch.sum(torch.nan_to_num(torch.stack([
            2*loss_l1(x, y),
            loss_cos(x, y, torch.ones(len(y)).to(x.device)),
        ])))

    def signal_encode(self, x, query, sampling_rate=65):
        # x: [bn, nvar, L]
        device = x.device

        # sensor encoding
        spec = self.sensor_model.calc_cwt(x, device=device).float()
        sensor_out = self.sensor_model.get_signal_embedding(spec, hidden_out=False, device=device) # bn, nvar, P, E

        if query.shape[0] == 1: # if single question for all samples in input batch
            query = query.expand(sensor_out.shape[0], query.shape[1]) # (bn, 2048)
        
        # aggregate
        bn, nvar, P, E  = sensor_out.shape

        query = query.unsqueeze(1).expand(bn, nvar*P, query.shape[1]) # (bn, nvar, 2048)

        # per channel aggregate
        ch_aggregate_out = self.aggregator(
            sensor_out, 
            query,
            device=device,
            rel_only=self.rel_only,
            use_query=self.use_query
        ) # bn*nvar, E

        return ch_aggregate_out

# ============= helper functions ================================================

def _resolve_normwear_path():
    env_path = os.getenv("NORMWEAR_PATH")
    if env_path:
        print()
        print(f"Using NormWear path from environment variable: {os.path.abspath(os.path.expanduser(env_path))}")
        print()
        return os.path.abspath(os.path.expanduser(env_path))
    return "mosaic-laboratory/normwear"

def load_model(model_name='normwear'):
    # all models should follows the function structure of AST_API
    if model_name == 'normwear':
        model = NormWearZeroShotHF(hf_model_id=_resolve_normwear_path())
    # elif model_name == 'clap':
    #     model = CLAP_API()
    else:
        print("Model not supported. ")
        exit()
    
    # return
    model = model.to(DEVICE)
    model.eval()

    # # check number of parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} Number of parameters: {total_params}")
    # exit()

    return model


def _resolve_dataset_root(ds_name):
    # resolve paths from this file location (instead of root_prefix)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)  # .../NormWear
    
    candidates = [
        os.path.normpath(os.path.join(project_root, "data", ds_name)),
        os.path.normpath(os.path.join(project_root, "data", os.path.basename(ds_name))),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]

def save_roc_curve_and_threshold_metrics(y_true, y_score, save_path):
            
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_pos = y_score[:, 1]
    else:
        y_pos = y_score.reshape(-1)

    #fpr, tpr, thresholds = roc_curve(y_true, y_pos)
    """
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()"""

    metrics_by_threshold = []
    thresholds = np.arange(0.0, 1.0, 0.05)  # Thresholds from 0 to 1 with a step of 0.05
    for thr in thresholds:
        y_pred = (y_pos >= thr).astype(int)
        metrics_by_threshold.append({
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0,average="macro")),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0,average="macro")),
        })

    return metrics_by_threshold
def zs_inference(
    ds_name="wearable_downstream/PPG_HTN", 
    model_name='normwear-msitf', 
    task_idx=0,
    batch_size=1024
):
    
    # construct model
    model = load_model(model_name=model_name)

    dataset_root = _resolve_dataset_root(ds_name)
    split = json.load(open(os.path.join(dataset_root, "train_test_split.json")))

    sample_root = os.path.join(dataset_root, "sample_for_downstream")

    # get embedding for each sample
    tasks_type = dict()
    embeds_all, labels_all = list(), dict()
    curr_samples, batch_num = list(), 0
    for fn in tqdm(sorted(os.listdir(sample_root))):
        # edge case
        if fn[0] == '.' or fn not in split['test']:
            continue
        
        # load sample
        # read data
        with open(os.path.join(sample_root, fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # expand 1 dimension if only single dimension
        if len(sample['data'].shape) == 1:
            sample['data'] = np.expand_dims(sample['data'], axis=0) # nvar, L
        
        # store for batch operation
        task_name = CLASS_NUM[ds_name]['names'][task_idx]
        if labels_all.get(task_name) is None:
            labels_all[task_name] = list()
            tasks_type[task_name] = [k for k in sample['label'][task_idx].keys()][0]
        labels_all[task_name].append(sample['label'][task_idx][tasks_type[task_name]])
        curr_samples.append(sample['data'])
        batch_num += 1
        if batch_num < batch_size:
            continue

        # batch operation

        # test clap pipeline
        with torch.no_grad():
            embeds_all += model(
                torch.from_numpy(np.stack(curr_samples)).float().to(DEVICE), 
                [(ds_name, CLASS_NUM[ds_name]['names'][task_idx])],
                label=None
            ).cpu().detach().numpy().tolist() # bn, E
        
        # refresh vars
        curr_samples, batch_num = list(), 0
    
    # calculate last batch
    if len(curr_samples) > 0:
        with torch.no_grad():
            embeds_all += model(
                torch.from_numpy(np.stack(curr_samples)).float().to(DEVICE), 
                [(ds_name, CLASS_NUM[ds_name]['names'][task_idx])],
                label=None
            ).cpu().detach().numpy().tolist() # bn, E
    
    # calculate predictions
    embeds_all = torch.tensor(embeds_all).to(DEVICE) # N, E

    for k in labels_all:
        # if has more than 1 dimensions
        if isinstance(labels_all[k][0], np.ndarray):
            if len(labels_all[k][0].shape) >= 1:
                labels_all[k] = [tuple(l) for l in labels_all[k]]
            else:
                labels_all[k] = [int(l) for l in labels_all[k]]
    
        # get choices
        label_name_map = [l for l in set(labels_all[k])]

        # text encoding
        choice_embeds = txt_encode(
            task=[(ds_name, CLASS_NUM[ds_name]['names'][task_idx])], 
            label=label_name_map, 
            model=model, 
            task_type=tasks_type[k]
        ) # num_label, E

        # label map, y_true, distance, task_type
        scores, y_probs = zs_evaluate(
            sensor_embeds=embeds_all, # tensor
            choice_embeds=choice_embeds, # tensor
            label_name_map=label_name_map, # dict
            task_type=tasks_type[k], # str
            y_trues=np.array(labels_all[k]) # np array
        )
        

        # expects zs_evaluate to also provide the class probabilities as y_probs
        # e.g. scores, y_probs = zs_evaluate(...)
        roc_save_path = os.path.join(dataset_root, f"{ds_name.replace('/', '_')}_{k}_roc.png")
        threshold_metrics = save_roc_curve_and_threshold_metrics(
            np.array(labels_all[k]),
            y_probs,
            roc_save_path,
        )
        scores = [round(s*100, 3) for s in scores]
        return threshold_metrics, scores
        

def zs_evaluate(
    sensor_embeds=None,
    choice_embeds=None,
    label_name_map=None,
    task_type=None,
    y_trues=None
):  
    # L1 distance
    distances = torch.abs(sensor_embeds[:, None, :] - choice_embeds[None, :, :]).sum(dim=-1) # bn, num_choice
    # distances = 1 / torch.matmul(sensor_embeds, choice_embeds.T)  # bn, num_choice
    # distances = distances + (0.5*dt_distances)

    # # check
    # print(distances.shape)
    # exit()

    if task_type == "reg":
        y_preds = np.array([label_name_map[idx] for idx in torch.argmin(distances, dim=1).cpu().numpy()]) # bn
        return [1 - np.mean(np.absolute(y_trues - y_preds) / y_trues)], y_preds
    else:
        sims = distances
        sims = 1 - (sims / torch.sum(sims, dim=1, keepdim=True))
        sims = torch.nan_to_num(sims) + 1e-8 # bn, num_choice
        
        y_preds = nn.functional.softmax(sims.float(), dim=1).detach().cpu().numpy() # bn, num_choice
    
        print("Classes in Test:", set(y_trues))
        if len(set(y_trues)) <= 2:
            return [roc_auc_score(y_trues, y_preds[:, 1])], y_preds
        else:
            # for i in range(len(y_trues)):
            #     print(y_trues[i], np.argmax(y_preds[i]))
            # print(y_trues, y_preds)
            return [roc_auc_score(y_trues, y_preds, multi_class="ovo", average="macro")], y_preds

if __name__ == '__main__':
    # python3 -m NormWear.zero_shot.zero_shot_inference normwear --dataset wesad
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', nargs='?', default='normwear')
    parser.add_argument('--dataset', default='all', help='Run only one dataset, e.g. wesad')
    parser.add_argument('--times', default='1', help='for how many times to run the evaluation, default 1')
    args = parser.parse_args()

    model_name = args.model_name

    # gc.collect()
    # torch.cuda.empty_cache()

    base_ds_names = [
        "wearable_downstream/PPG_HTN",
        "wearable_downstream/PPG_DM",
        "wearable_downstream/PPG_CVA",
        "wearable_downstream/PPG_CVD",
        "wearable_downstream/non_invasive_bp", 
        "wearable_downstream/ppg_hgb", 
        "wearable_downstream/indian-fPCG",
        "wearable_downstream/ecg_heart_cat", # **
        "wearable_downstream/drive_fatigue", # *
        "wearable_downstream/gameemo", # **
        "wearable_downstream/uci_har", # ***
        "wearable_downstream/wesad", # ***
        "wearable_downstream/emg-tfc",
        "wearable_downstream/Epilepsy",
    ]

    if args.dataset != 'all':
        base_ds_names = [ds for ds in base_ds_names if ds.endswith(f"/{args.dataset}")]

    ds_names = list(base_ds_names)

    for d_i in range(len(ds_names)):
        task_idx = 0
        if ds_names[d_i] == "wearable_downstream/Epilepsy":
            ds_names[d_i] = (ds_names[d_i], 0)
            for d_j in range(1, 5):
                ds_names.append(("wearable_downstream/Epilepsy", d_j))
        else:
            ds_names[d_i] = (ds_names[d_i], 0)
    # launch zero shot evaluation
    for ds in ds_names:
        print(ds)
        ds_name, task_idx = ds
        eval_metrics, scores= [], []
        
        for t in tqdm(range(int(args.times)), desc=f"Processing {ds_name}"):
            print(f"Run {t+1}/{args.times} for {ds_name}...")
            eval, score = zs_inference(
                ds_name=ds_name, 
                model_name=model_name, 
                task_idx=task_idx,
                batch_size=64
            )
            eval_metrics.append(eval)
            scores.append(score)
        print(f"Final evaluation metrics for {ds_name}:", scores)
        print(f"Final evaluation metrics for {ds_name}:", eval_metrics)