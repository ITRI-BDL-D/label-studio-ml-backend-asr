from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from openai import AzureOpenAI

import os
import json
from uuid import uuid4
import requests
import concurrent.futures
import shutil
from tqdm import tqdm

from utils import wav2segments, any2wav


BDLAI_EU2_client = AzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version="2024-02-01",
)


def asr_worker(speech_input):
    """
    Read WAVE from `audio`
    Put results from ASR server to `text`
    """
    audio = speech_input['audio']
    config_data = speech_input['config']
    url = speech_input['url']
    


    with open(audio, 'rb') as audio_file:
        files = {'data': audio_file}
        req = requests.post(url, files=files, data=config_data)

    try:
        info = json.loads(req.text)
        text = info['result']
        status = "200"
    except:
        text = "<fail to get response from asr server>"
        status = "-1"
    
    # print(f'{audio}: {text}')
    speech_input['text'] = text
    speech_input['status'] = status
    return speech_input


def asr_master(input_batch, num_threads=1):
    """
    Distribute inputs to workers.
    """
    output_batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_input = {executor.submit(asr_worker, input_data):input_data for input_data in input_batch}

        for future in concurrent.futures.as_completed(future_to_input):
            data = future.result()
            output_batch.append(data)
    return output_batch


def speech_to_text(speech, lang='zh'):
    # create temp folder
    base_name = os.path.basename(speech).split(".")[0]
    tmp_uid = str(uuid4())[:8] + "-" + base_name
    temp_dir = "tmp/{}".format(tmp_uid)
    os.makedirs(temp_dir)

    # WAV: PCM, 16k, 16bit
    std_audio = os.path.join(temp_dir, "std.wav")
    rtn = any2wav(speech, std_audio)
    if(rtn):
        print(f"{tmp_uid}: WAV ok.")
        print(std_audio)
    else:
        shutil.rmtree(temp_dir)
        return False

    # VAD: detect speech segments

    timestamps = wav2segments(std_audio, outputdir=temp_dir, mode=2)
    print("{}: VAD ok.".format(tmp_uid))

    print(f"{tmp_uid}: VAD ok.")
    asr_url = ''

    for ts in tqdm(timestamps, total=len(timestamps)):
        ts["url"] = asr_url
        ts["audio"] = ts["speech"]
        ts["config"] = {"language": lang}

    results = asr_master(timestamps, num_threads=4)
    results = sorted(results, key=lambda x: x["id"])
   
    shutil.rmtree(temp_dir)
    return results

class AOAIAsr(LabelStudioMLBase):
    """Custom ML Backend model"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}
        # Extra params: {self.extra_params}''')
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            "TextArea", "Audio"
        )
        from_name_label, to_name_label, value_label = self.label_interface.get_first_tag_occurence(
            "Labels", "Audio"
        )
        print(from_name, to_name, value)
        print(from_name_label, to_name_label, value_label)

        audio_paths = []
        for task in tasks:
            audio_url = task["data"].get(value) or task["data"].get(DATA_UNDEFINED_NAME)
            audio_path = get_local_path(audio_url, task_id=task.get("id"))
            audio_paths.append(audio_path)

        predictions = []
        for audio_file in audio_paths:
            res = speech_to_text(audio_file)
            # print(res)
            # transcription = BDLAI_EU2_client.audio.transcriptions.create(
            #     model="whisper",
            #     file=open(audio_file, "rb"),
            #     language='zh',
            #     response_format="verbose_json",
            #     timestamp_granularities=["segment"],
            # )
            # print(transcription.text)
            results = []
            # for seg in transcription.segments:
                # results.append(
                #     {
                #         "id": f'id_{seg.id}',
                #         "from_name": from_name_label,
                #         "to_name": to_name,
                #         "type": "labels",
                #         "value": {
                #             "start": seg.start,
                #             "end": seg.end,
                #             "labels": ["Speech"],
                #         },
                #     }
                # )
                # results.append(
                #     {
                #         "id": f'id_{seg.id}',
                #         "from_name": from_name,
                #         "to_name": to_name,
                #         "type": "textarea",
                #         "value": {
                #             "start": seg.start,
                #             "end": seg.end,
                #             "text": [seg.text],
                #         },
                #     }
                # )
            for seg in res: #transcription.segments
                # print(f"Start:{seg.start} End:{seg.end} Text:{seg.text}")
                results.append(
                    {
                        "id": f"id_{seg['id']}",
                        "from_name": from_name_label,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": seg['start'],
                            "end": seg['stop'],
                            "labels": ["Speech"],
                        },
                    }
                )
                results.append(
                    {
                        "id": f"id_{seg['id']}",
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "textarea",
                        "value": {
                            "start": seg['start'],
                            "end": seg['stop'],
                            "text": [seg['text']],
                        },
                    }
                )
            predictions.append(
                {
                    "result": results,
                    "score": 1.0,
                    "model_version": self.get("model_version"),
                }
            )

        return ModelResponse(predictions=predictions) # predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
