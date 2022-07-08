# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import re
import sys
import traceback
from typing import Union

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.restful.request import TTSRequest
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.restful.response import TTSResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/tts/help')
def help():
    """help

    Returns:
        json: [description]
    """
    response = {
        "success": "True",
        "code": 200,
        "message": {
            "global": "success"
        },
        "result": {
            "description": "tts server",
            "text": "sentence to be synthesized",
            "audio": "the base64 of audio"
        }
    }
    return response


@router.get("/paddlespeech/tts")
def tts_get(text: str,
            spk_id: int = 0,
            speed: float = 1.0,
            volume: float = 1.0,
            sample_rate: int = 0,
            save_path: str = None,
            range: Union[str, None] = Header(default=None),
            ):
    # Check parameters
    if speed <= 0 or speed > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid speed value, the value should be between 0 and 3.")
    if volume <= 0 or volume > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid volume value, the value should be between 0 and 3.")
    if sample_rate not in [0, 16000, 8000]:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid sample_rate value, the choice of value is 0, 8000, 16000.")
    if save_path is not None and not save_path.endswith(
            "pcm") and not save_path.endswith("wav"):
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid save_path, saved audio formats support pcm and wav")

    # run
    try:
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']
        logger.info("Get tts engine successfully.")

        if tts_engine.engine_type == "python":
            from paddlespeech.server.engine.tts.python.tts_engine import PaddleTTSConnectionHandler
        elif tts_engine.engine_type == "inference":
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import PaddleTTSConnectionHandler
        else:
            logger.error("Offline tts engine only support python or inference.")
            sys.exit(-1)

        connection_handler = PaddleTTSConnectionHandler(tts_engine)
        lang, target_sample_rate, duration, wav_base64 = connection_handler.run(
            text, spk_id, speed, volume, sample_rate, save_path)
        data_bytes = base64.b64decode(wav_base64)
        size = len(data_bytes)
        status_code = 200
        headers = {}
        if range:
            HTTP_RANGE_HEADER = re.compile(r'bytes=([0-9]+)\-(([0-9]+)?)')
            m = re.match(HTTP_RANGE_HEADER, range)
            if m:
                start_str = m.group(1)
                start = int(start_str)
                end_str = m.group(2)
                end = -1
                # end存在
                if len(end_str) > 0:
                    end = int(end_str)
                # range存在时，让请求支持断点续传,status_code改为206
                status_code = 206
                if end == -1:
                    # 此处的size是文件大小
                    headers["Content-Length"] = str(size - start)
                else:
                    # Content-Length也要改变
                    headers["Content-Length"] = str(end - start + 1)
                headers["Accept-Ranges"] = "bytes"
                if end < 0:
                    content_range_header_value = "bytes %d-%d/%d" % (
                        start, size - 1, size)
                    data_bytes = data_bytes[start:]
                else:
                    content_range_header_value = "bytes %d-%d/%d" % (
                        start, end, size)
                    data_bytes = data_bytes[start:end + 1]
                headers["Content-Range"] = content_range_header_value
                headers["Connection"] = "keep-alive"
        buf = io.BytesIO(data_bytes)
        response = StreamingResponse(
            buf, status_code=status_code, headers=headers, media_type="audio/wav")
    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response


@router.post(
    "/paddlespeech/tts", response_model=Union[TTSResponse, ErrorResponse])
def tts(request_body: TTSRequest):
    """tts api

    Args:
        request_body (TTSRequest): [description]

    Returns:
        json: [description]
    """

    logger.info("request: {}".format(request_body))

    # get params
    text = request_body.text
    spk_id = request_body.spk_id
    speed = request_body.speed
    volume = request_body.volume
    sample_rate = request_body.sample_rate
    save_path = request_body.save_path

    # Check parameters
    if speed <= 0 or speed > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid speed value, the value should be between 0 and 3.")
    if volume <= 0 or volume > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid volume value, the value should be between 0 and 3.")
    if sample_rate not in [0, 16000, 8000]:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid sample_rate value, the choice of value is 0, 8000, 16000.")
    if save_path is not None and not save_path.endswith(
            "pcm") and not save_path.endswith("wav"):
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid save_path, saved audio formats support pcm and wav")

    # run
    try:
        # get single engine from engine pool
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']
        logger.info("Get tts engine successfully.")

        if tts_engine.engine_type == "python":
            from paddlespeech.server.engine.tts.python.tts_engine import PaddleTTSConnectionHandler
        elif tts_engine.engine_type == "inference":
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import PaddleTTSConnectionHandler
        else:
            logger.error("Offline tts engine only support python or inference.")
            sys.exit(-1)

        connection_handler = PaddleTTSConnectionHandler(tts_engine)
        lang, target_sample_rate, duration, wav_base64 = connection_handler.run(
            text, spk_id, speed, volume, sample_rate, save_path)

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success."
            },
            "result": {
                "lang": lang,
                "spk_id": spk_id,
                "speed": speed,
                "volume": volume,
                "sample_rate": target_sample_rate,
                "duration": duration,
                "save_path": save_path,
                "audio": wav_base64
            }
        }
    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response


@router.post("/paddlespeech/tts/streaming")
async def stream_tts(request_body: TTSRequest):
    # get params
    text = request_body.text
    spk_id = request_body.spk_id

    engine_pool = get_engine_pool()
    tts_engine = engine_pool['tts']
    logger.info("Get tts engine successfully.")

    if tts_engine.engine_type == "online":
        from paddlespeech.server.engine.tts.online.python.tts_engine import PaddleTTSConnectionHandler
    elif tts_engine.engine_type == "online-onnx":
        from paddlespeech.server.engine.tts.online.onnx.tts_engine import PaddleTTSConnectionHandler
    else:
        logger.error("Online tts engine only support online or online-onnx.")
        sys.exit(-1)

    connection_handler = PaddleTTSConnectionHandler(tts_engine)

    return StreamingResponse(
        connection_handler.run(sentence=text, spk_id=spk_id))


@router.get("/paddlespeech/tts/streaming/samplerate")
def get_samplerate():
    try:
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']
        logger.info("Get tts engine successfully.")
        sample_rate = tts_engine.sample_rate

        response = {"sample_rate": sample_rate}

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
