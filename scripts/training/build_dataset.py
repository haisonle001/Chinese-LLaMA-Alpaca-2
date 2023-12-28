import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers


IGNORE_INDEX = -100

logger = logging.getLogger('__name__')


PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. Write a response based on the instruction\n"
        "<</SYS>>\n\n ### Instruction:\n{instruction}\n\n### Response: [/INST] "
    )

PROMPT_TEMPLATE_INPUT = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. Write a response based on the instruction\n"
        "<</SYS>>\n\n ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: [/INST] "
    )

### only
PROMPT_CHAT=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n\n"
        "<</SYS>>\n\n"
        "{user_message} [/INST] {model_reply}</s>")


### chat-version
# PROMPT_TEMPLATE = (
#         "[INST] <<SYS>>\n"
#         "You are a helpful assistant.\n"
#         "<</SYS>>\n\n{instruction}\n\n [/INST] "
#     )

# PROMPT_TEMPLATE_INPUT = (
#         "[INST] <<SYS>>\n"
#         "You are a helpful assistant. Write a response based on the instruction\n"
#         "<</SYS>>\n\n{instruction}. {input}\n\n [/INST] "
#     )


def prompt_chat(temp_dict):
    temp_dict= temp_dict['conversations']
    prompt_chat= PROMPT_CHAT.format_map({'user_message': temp_dict[0]['value'],'model_reply': temp_dict[1]['value']})
    for temp in temp_dict[1:-1]:
        # if temp['from']=='user':
        prompt_chat+= f"<s>[INST] {temp['value']} [/INST]</s>"
    prompt_chat+=f"<s>[INST] {temp_dict[-1]['value']} [/INST]"
    return prompt_chat

# PROMPT_TEMPLATE=("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:")

# PROMPT_TEMPLATE=('<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{instruction}\n\n{input} [/INST]')
# PROMPT_TEMPLATE=('<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{instruction} [/INST]')

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []

        for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
            if input =="" and instruction=="":
                source=""
                output= output
            elif input is not None and input !="":
                prompt =  PROMPT_TEMPLATE_INPUT
                source = prompt.format_map({'instruction':instruction,'input':input})
            else: 
                prompt = PROMPT_TEMPLATE
                source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"
            print(source+target)
            # print(source)
            # print(target)
            # sources.append(source)
            # targets.append(target)
            # print(source)
            # print(target)
            sources.append('')
            targets.append(source+target)
            # break
        # print(sources[:2],targets[:2])
        
        ### Phase1 
        # for instruction, output in zip(examples['translated_question'],examples['translated_answer']):
        #     # if input is not None and input !="":
        #     #     prompt = PROMPT_TEMPLATE_INPUT
        #     #     # instruction = instruction+'\n'+input
        #     #     source = prompt.format_map({'instruction':instruction,'input':input})
        #     # else:
        #     prompt = PROMPT_TEMPLATE
        #     source = prompt.format_map({'instruction':instruction})
        #     # source = prompt.format_map({'instruction':instruction,'input':input})
        #     # source = prompt.format_map({'instruction':instruction})
        #     target = f"{output}{tokenizer.eos_token}"
        #     # print(source)
        #     # print(target)
        #     sources.append(source)
        #     targets.append(target)
        #     # break
        # for question,input, output,explanation in zip(examples['question'],examples['choices'],examples['answer'],examples['explanation']):
            # # if input is not None and input !="":
            # #     prompt = PROMPT_TEMPLATE_INPUT
            # #     # instruction = instruction+'\n'+input
            # #     source = prompt.format_map({'instruction':instruction,'input':input})
            # # else:
            # prompt = PROMPT_TEMPLATE
            # source = prompt.format_map({'instruction': question + '\n'+ '\n'.join(input)})
            # # source = prompt.format_map({'instruction':instruction,'input':input})
            # # source = prompt.format_map({'instruction':instruction})
            # target = f"{explanation}\nĐáp án đúng là:\n{output}{tokenizer.eos_token}"
            # # print(source)
            # # print(target)
            # sources.append(source)
            # targets.append(target)
            # # break
            
        # # ### CODELLM
        # for test_api, test_testplan, test_bdd in zip(examples['test_api'],examples['test_testplan'],examples['test_bdd']):
        #     if  test_bdd =="":
        #         prompt = PROMPT_TEMPLATE_PHASE_1
        #         # instruction = instruction+'\n'+input
        #         source = prompt.format_map({'test_api':test_api})
                
        #         target = f"Testing Plan:\n{test_testplan}{tokenizer.eos_token}"
        #         sources.append(source)
        #         targets.append(target)
        #     else:
        #         continue
        #         # prompt = PROMPT_TEMPLATE_PHASE_2
        #         # source = prompt.format_map({'test_api':test_api,'test_testplan':test_testplan})
                
        #         # target = f"Cucumber tests:\n{test_bdd}{tokenizer.eos_token}"
        #         # sources.append(source)
        #         # targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0]+f"_{max_seq_length}")
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],

            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
