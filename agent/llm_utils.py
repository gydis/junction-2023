from __future__ import annotations

import json

from fastapi import WebSocket
import time
import os

import openai
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from langchain.llms import HuggingFaceHub
from langchain.schema import ChatMessage
from langchain.chains import LLMChain
from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from agent.prompts import auto_agent_instructions
from config import Config

CFG = Config()

openai.api_key = CFG.openai_api_key

from typing import Optional
import logging

def create_chat_completion(
    messages: list,  # type: ignore
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 2000,
    stream: Optional[bool] = False,
    websocket: WebSocket | None = None,
    repetition_penalty = None,
    top_p = None,
    summ = False,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    Returns:
        str: The response from the chat completion
    """

    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 8001:
        raise ValueError(f"Max tokens cannot be more than 8001, but got {max_tokens}")
    if stream and websocket is None:
        raise ValueError("Websocket cannot be None when stream is True")

    # create response
    for attempt in range(10):  # maximum of 10 attempts
        print(f"Stream: {stream}")
        response = send_chat_completion_request(
            messages, model, temperature, max_tokens, stream, websocket, repetition_penalty, top_p, summ
        )
        if response is not None: return response

    logging.error("Failed to get response from OpenAI API")
    raise RuntimeError("Failed to get response from OpenAI API")

def send_chat_completion_request(
    messages, model, temperature, max_tokens, stream, websocket, repetition_penalty, top_p, summ
):
    messages = [ChatMessage(content=e['content'], role=e['role']) for e in messages]
    if not stream:
        chat = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={'temperatue': temperature, 'max_new_tokens': max_tokens, 'repetition_penalty': repetition_penalty, 'top_p': top_p})
        summc = HuggingFaceHub(repo_id="facebook/bart-large-cnn")
        try:
            results = chat.invoke(messages) if not summ else summc.invoke(messages)
            # print(results)
        except Exception as e:
            print(f"{Fore.RED}Error in querying Azure: {e}{Style.RESET_ALL}")
            results = None
        # result = lc_openai.ChatCompletion.create(
        #     model=model, # Change model here to use different models
        #     messages=messages,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     provider=CFG.llm_provider, # Change provider here to use a different API
        # )
        # return result["choices"][0]["message"]["content"]
        return results


def choose_agent(task: str) -> dict:
    """Determines what agent should be used
    Args:
        task (str): The research question the user asked
    Returns:
        agent - The agent that will be used
        agent_role_prompt (str): The prompt for the agent
    """
    # try:
    #     response = create_chat_completion(
    #         model=CFG.smart_llm_model,
    #         messages=[
    #             {"role": "system", "content": f"{auto_agent_instructions()}"},
    #             {"role": "user", "content": f"task: {task}"}],
    #         temperature=0,
    #     )
    #     raise Exception("WIP")
    #     return response
    # except Exception as e:
    #     print(f"{Fore.RED}Error in choose_agent: {e}{Style.RESET_ALL}")
    return {"agent": "Default Agent",
            "agent_role_prompt": "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."}


