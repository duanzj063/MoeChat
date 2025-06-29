from utilss import config as CConfig
import requests
import json
import time
import asyncio
from threading import Thread
import base64
from fastapi.responses import JSONResponse
from io import BytesIO
from pydantic import BaseModel
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utilss.sv import SV
from utilss.agent import Agent
import re
import jionlp


if CConfig.config["Agent"]["is_up"]:
    agent = Agent()

try:
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        device="cuda:0",
    )
except:
    print("[提示]未安装ASR模型，开始自动安装ASR模型。")
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        model_id="iic/SenseVoiceSmall",
        local_dir="./utilss/models/SenseVoiceSmall",
        revision="master"
    )
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        # device="cuda:0",
        device="cpu",
    )

# 载入声纹识别模型
sv_pipeline = ""
if CConfig.config["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(CConfig.config["Core"]["sv"])
    is_sv = True
else:
    is_sv = False

# 提交到大模型
def to_llm(msg: list, res_msg_list: list, full_msg: list):
    def get_emotion(msg: str):
        res = re.findall(r'\[(.*?)\]', msg)
        if len(res) > 0:
            match = res[-1]
            if match and CConfig.config["extra_ref_audio"]:
                if match in CConfig.config["extra_ref_audio"]:
                    return match
    # def add_msg(msg: str):
    key = CConfig.config["LLM"]["key"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    data = {
        "model": CConfig.config["LLM"]["model"] ,
        "stream": True
    }
    if CConfig.config["LLM"]["extra_config"]:
        data.update(CConfig.config["LLM"]["extra_config"])
    data["messages"] = msg

    t_t = time.time()
    try:
        response = requests.post(url = CConfig.config["LLM"]["api"], json=data, headers=headers,stream=True)
    except:
        print("无法链接到LLM服务器")
        return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})
    
    # 信息处理
    # biao_dian_2 = ["…", "~", "～", "。", "？", "！", "?", "!"]
    biao_dian_3 = ["…", "~", "～", "。", "？", "！", "?", "!", ",", "，"]
    biao_dian_4 = ["…", "~", "～",  ",", "，"]

    res_msg = ""
    tmp_msg = ""
    j = True
    j2 = True
    ref_audio = ""
    ref_text = ""
    # biao_tmp = biao_dian_3
    for line in response.iter_lines():
        if line:
            try:
                if j:
                    print(f"\n[大模型延迟]{time.time() - t_t}")
                    t_t = time.time()
                    j = False
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str:
                        msg_t = json.loads(data_str)["choices"][0]["delta"]["content"]
                        res_msg += msg_t
                        tmp_msg += msg_t
                res_msg = res_msg.replace("...", "…")
                tmp_msg = tmp_msg.replace("...", "…")
            except:
                err = line.decode("utf-8")
                print(f"[错误]：{err}")
                continue
            # if not tmp_msg:
            #     continue
            ress = ""
            stat = 0
            for ii in range(len(tmp_msg)):
                if tmp_msg[ii] in ["(", "（", "[", "{"]:
                    stat += 1
                    continue
                if tmp_msg[ii] in [")", "）", "]", "}"]:
                    stat -= 1
                    continue
                if stat != 0:
                    continue
                if tmp_msg[ii] not in biao_dian_3:
                    continue
                if (tmp_msg[ii] in biao_dian_4) and j2 == False and len(re.sub(r'[$(（[].*?[]）)]', '', tmp_msg[:ii+1])) <= 10:
                    continue

                # 提取文本中的情绪标签，并设置参考音频
                emotion = get_emotion(tmp_msg)
                if emotion:
                    if emotion in CConfig.config["extra_ref_audio"]:
                        ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                        ref_text = CConfig.config["extra_ref_audio"][emotion][1]
                ress = tmp_msg[:ii+1]
                ress = jionlp.remove_html_tag(ress)
                ttt = ress
                if j2:
                    print(f"\n[开始合成首句语音]{time.time() - t_t}")
                    for i in range(len(ress)):
                        if ress[i] == "\n" or ress[i] == " ":
                            try:
                                ttt = ress[i+1:]
                            except:
                                ttt = ""
                # if not j2:
                #     if len(re.sub(r'[$(（[].*?[]）)]', '', ttt)) < 4:
                #         continue
                if ttt:
                    res_msg_list.append([ref_audio, ref_text, ttt])
                # print(f"[合成文本]{ress}")
                if j2:
                    j2 = False
                try:
                    tmp_msg = tmp_msg[ii+1:]
                except:
                    tmp_msg = ""
                break


    if len(tmp_msg) > 0:
        emotion = get_emotion(tmp_msg)
        if emotion:
            if emotion in CConfig.config["extra_ref_audio"]:
                ref_audio = CConfig.config["extra_ref_audio"][emotion][0]
                ref_text = CConfig.config["extra_ref_audio"][emotion][1]
        res_msg_list.append([ref_audio, ref_text, tmp_msg])

    # 返回完整上下文 
    res_msg = jionlp.remove_html_tag(res_msg)
    if len(res_msg) == 0:
        full_msg.append(res_msg)
        res_msg_list.append("DONE_DONE")
        return
    ttt = ""
    for i in range(len(res_msg)):
        if res_msg[i] != "\n" and res_msg[i] != " ":
            ttt = res_msg[i:]
            break
            
    full_msg.append(ttt)
    print(full_msg)
    # print(res_msg_list)
    res_msg_list.append("DONE_DONE")

def tts(datas: dict):
    res = requests.post(CConfig.config["GSV"]["api"], json=datas, timeout=10)
    if res.status_code == 200:
        return res.content
    else:
        print(f"[错误]tts语音合成失败！！！")
        print(datas)
        return None
    

def clear_text(msg: str):
    msg = re.sub(r'\{(image|meme|pics):.*?\}', '', msg) # 新增：移除所有image和meme标签
    msg = re.sub(r'[$(（[].*?[]）)]', '', msg)
    msg = msg.replace(" ", "").replace("\n", "")
    tmp_msg = ""
    biao = ["…", "~", "～", "。", "？", "！", "?", "!",  ",",  "，"]
    for i in range(len(msg)):
        if msg[i] not in biao:
          tmp_msg = msg[i:]
          break
    # msg = jionlp.remove_exception_char(msg)
    return tmp_msg


# TTS并写入队
def to_tts(tts_data: list):
    # def is_punctuation(char):
    #     return unicodedata.category(char).startswith('P')
    msg = clear_text(tts_data[2])
    # print(f"[实际输入文本]{tts_data[2]}[tts文本]{msg}")
    if len(msg) == 0:
        return "None"
    ref_audio = tts_data[0]
    ref_text = tts_data[1]
    datas = {
        "text": msg,
        "text_lang": CConfig.config["GSV"]["text_lang"],
        "ref_audio_path": CConfig.config["GSV"]["ref_audio_path"],
        "prompt_text": CConfig.config["GSV"]["prompt_text"],   
        "prompt_lang": CConfig.config["GSV"]["prompt_lang"],
        "seed": CConfig.config["GSV"]["seed"],
        "top_k": CConfig.config["GSV"]["top_k"],
        "batch_size": CConfig.config["GSV"]["batch_size"],
    }
    if CConfig.config["GSV"]["ex_config"]:
        for key in CConfig.config["GSV"]["ex_config"]:
            datas[key] = CConfig.config["GSV"]["ex_config"][key]
    if ref_audio:
        datas["ref_audio_path"] = ref_audio
        datas["prompt_text"] = ref_text
    try:
        byte_data = tts(datas)
        audio_b64 = base64.urlsafe_b64encode(byte_data).decode("utf-8")
        return audio_b64
    except:
        return "None"

def ttts(res_list: list, audio_list: list):
    i = 0
    while True:
        if i < len(res_list):
            if res_list[i] == "DONE_DONE":
                audio_list.append("DONE_DONE")
                print(f"完成...")
                break
            # t_t = time.time()
            audio_list.append(to_tts(res_list[i]))
            # if i == 0:
            #     print(f"\n[首句语音耗时]{time.time() - t_t}")
            i += 1
        time.sleep(0.05)


# asr功能
def asr(params: str):
    global asr_model
    global is_sv
    global sv_pipeline

    audio_data = base64.urlsafe_b64decode(params.encode("utf-8"))

    tt = time.time()
    if is_sv:
        if not sv_pipeline.check_speaker(audio_data):
            return None
    # with open(f"./tmp/{tt}.wav", "wb") as file:
    #     file.write(audio_data)
    audio_data = BytesIO(audio_data)
    res = asr_model.generate(
        input=audio_data,
        # input=f"{model.model_path}/example/zh.mp3",
        cache={},
        language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
        ban_emo_unk=True,
        use_itn=False,
        # batch_size=200,
    )
    # print(f"{model.model_path}/example/zh.mp3",)
    text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")
    # text = res[0]["text"]
    print()
    print(f"[{time.time() - tt}]{text}\n\n")
    if text:
        return text
    return None


class tts_data(BaseModel):
    msg: list

async def text_llm_tts(params: tts_data, start_time):
        # print(params)
        res_list = []
        audio_list = []
        full_msg = []
        if CConfig.config["Agent"]["is_up"]:
            global agent
            t = time.time()
            msg_list = agent.get_msg_data(params.msg[-1]["content"])
            print(f"[提示]获取上下文耗时：{time.time() - t}")
        else:
            msg_list = params.msg
        llm_t = Thread(target=to_llm, args=(msg_list, res_list, full_msg, ))
        llm_t.daemon = True
        llm_t.start()
        tts_t = Thread(target=ttts, args=(res_list, audio_list, ))
        tts_t.daemon = True
        tts_t.start()

        i = 0
        stat = True
        while True:
            if i < len(audio_list):
                if audio_list[i] == "DONE_DONE":
                    data = {"file": None, "message": full_msg[0], "done": True}
                    if CConfig.config["Agent"]["is_up"]:    # 刷新智能体上下文内容
                        agent.add_msg(re.sub(r'<.*?>', '', full_msg[0]).strip())
                    yield f"data: {json.dumps(data)}\n\n"
                data = {"file": audio_list[i], "message": res_list[i][2], "done": False}
                # audio = str(audio_list[i])
                # yield str(data)
                if stat:
                    print(f"\n[服务端首句处理耗时]{time.time() - start_time}\n")
                    stat = False
                yield f"data: {json.dumps(data)}\n\n"
                i += 1
            await asyncio.sleep(0.05)
