# Description: Research assistant class that handles the research process for a given question.

# libraries
import asyncio
import json
import hashlib

from actions.web_search import web_search
from actions.web_scrape import async_browse
from processing.text import \
    write_to_file, \
    create_message, \
    create_chat_completion, \
    read_txt_files, \
    write_md_to_pdf
from config import Config
from agent import prompts
import os
import string
import re


CFG = Config()


class ResearchAgent:
    def __init__(self, question, agent, agent_role_prompt, workload, websocket=None):
        """ Initializes the research assistant with the given question.
        Args: question (str): The question to research
        Returns: None
        """
        
        self.workload = workload
        self.question = question
        self.agent = agent
        self.agent_role_prompt = agent_role_prompt if agent_role_prompt else prompts.generate_agent_role_prompt(agent)
        self.visited_urls = set()
        self.research_summary = ""
        self.dir_path = f"./outputs/{hashlib.sha1(question.encode()).hexdigest()}"
        self.websocket = websocket

    async def stream_output(self, output):
        if not self.websocket:
            return print(output)
        await self.websocket.send_json({"type": "logs", "output": output})

    async def get_new_urls(self, url_set_input):
        """ Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.visited_urls:
                await self.stream_output(f"✅ Adding source url to research: {url}\n")

                self.visited_urls.add(url)
                new_urls.append(url)

        return new_urls

    async def call_agent(self, action, stream=False, websocket=None, w=None):
        messages = [{
            "role": "system",
            "content": self.agent_role_prompt
        }, {
            "role": "user",
            "content": action,
        }]
        if w is not None:
            answer = create_chat_completion(
                model=CFG.smart_llm_model,
                messages=messages,
                stream=stream,
                websocket=websocket,
                max_tokens=500,
            )
        else:
            answer = create_chat_completion(
                model=CFG.smart_llm_model,
                messages=messages,
                stream=stream,
                websocket=websocket,
            )
            
        return answer.content

    async def create_search_queries(self):
        """ Creates the search queries for the given question.
        Args: None
        Returns: list[str]: The search queries for the given question
        """
        result = await self.call_agent(prompts.generate_search_queries_prompt(self.question, self.workload))
        # await self.stream_output(f"🔎 I got this shit: {result}...")
        result = re.compile("\"[^\"]*\"").findall(result)
        result = [s[1:-2] for s in result]
        result = result[:int(self.workload)]
        await self.stream_output(f"🧠 I will conduct my research based on the following queries: {result}...")
        return result

    async def async_search(self, query):
        """ Runs the async search for the given query.
        Args: query (str): The query to run the async search for
        Returns: list[str]: The async search for the given query
        """
        search_results = json.loads(web_search(query))
        new_search_urls = self.get_new_urls([url.get("href") for url in search_results])

        await self.stream_output(f"🌐 Browsing the following sites for relevant information: {new_search_urls}...")

        # Create a list to hold the coroutine objects
        tasks = [async_browse(url, query, self.websocket) for url in await new_search_urls]

        # Gather the results as they become available
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        return responses

    async def run_search_summary(self, query):
        """ Runs the search summary for the given query.
        Args: query (str): The query to run the search summary for
        Returns: str: The search summary for the given query
        """

        await self.stream_output(f"🔎 Running research for '{query}'...")
        responses = await self.async_search(query)

        result = "\n".join(responses)
        os.makedirs(os.path.dirname(f"{self.dir_path}/research-{query}.txt"), exist_ok=True)
        write_to_file(f"{self.dir_path}/research-{query}.txt", result)
        return result

    async def conduct_research(self):
        """ Conducts the research for the given question.
        Args: None
        Returns: str: The research for the given question
        """
        self.research_summary = read_txt_files(self.dir_path) if os.path.isdir(self.dir_path) else ""

        if not self.research_summary:
            search_queries = await self.create_search_queries()
            for query in search_queries:
                research_result = await self.run_search_summary(query)
                self.research_summary += f"{research_result}\n\n"

        await self.stream_output(f"Total research words: {len(self.research_summary.split(' '))}")

        return self.research_summary

    async def write_report(self, report_type, websocket=None):
        """ Writes the report for the given question.
        Args: None
        Returns: str: The report for the given question
        """
        report_type_func = prompts.get_report_by_type(report_type)
        await self.stream_output(f"✍️ Writing {report_type} for research task: {self.question}...")

        print(f"Prompt for the final report:\n{report_type_func(self.question, self.research_summary)}")
        answer = await self.call_agent(report_type_func(self.question, self.research_summary),
                                       stream=False, websocket=websocket, w=True)
        # if websocket is True than we are streaming gpt response, so we need to wait for the final response
        # print(f"Final answer from the model: {answer}")
        final_report = answer

        path = await write_md_to_pdf(report_type, self.dir_path, final_report)
        await self.stream_output(f"📝 Final report:\n {final_report}")

        return answer, path