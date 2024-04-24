from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
from langchain.embeddings import LlamaCppEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import pandas as pd

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from langchain.document_loaders import DirectoryLoader
import tqdm
import requests



class HznGenAI:
	def __init__(self, model_name = "TheBloke/Llama-2-13B-GPTQ", model_basename = "model",use_triton = False, cache_dir = 'models', 
				open_ai_key = None, needs_quantized_loading = True):
		self.model_name = model_name
		self.model_basename = model_basename
		self.use_triton = use_triton
		self.cache_dir = cache_dir
		self.needs_quantized_loading = needs_quantized_loading
		self.model_load_flag = False
		self.vectorstore = None
		self.qa_chain_flag = False
		self.open_ai_model_flag = False
		self.embedding_model_flag = False
		self.prompt_template = DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

	def load_text_generation_model(self, max_new_tokens = 512):
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = True)
		if self.needs_quantized_loading == True:
			self.model_obj = AutoGPTQForCausalLM.from_quantized(self.model_name,
														model_basename=self.model_basename,
        													use_safetensors=True,
        													trust_remote_code=True,
													        device="cuda:0",
													        use_triton=self.use_triton,
													        quantize_config=None,
													        cache_dir = self.cache_dir)
		else:
			self.model_obj = AutoModelForCausalLM.from_pretrained(self.model_name,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")
		self.pipe = pipeline(
    "text-generation",
    model=self.model_obj,
    tokenizer=self.tokenizer,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

		self.model_load_flag = True

	def execute_prompt(self,user_input):
		if self.load_text_generation_model == False:
			print('Model Not Yet Loaded, Loading...')
			self.load_model()

		prompt = self._create_prompt(user_input)
		output = self.pipe(prompt)[0]['generated_text']
		output = output.replace(prompt,"")
		return output

	def _create_prompt(self, user_input):
		return self.prompt_template.format(user_input)

	def create_vectordb(self,document_directory,vectordb_name,load = True):
		if self.embedding_model_flag == False:
			"No Embedding Model Loaded"
		loader = DirectoryLoader(document_directory)
		pages = loader.load_and_split()

		vectorstore = FAISS.from_documents(pages,self.embedding_model)
		vectorstore.save_local(vectordb_name)
		if load:
			self.vectorstore = vectorstore
			
	def load_in_vectordb(self,vectordb_name):
		if self.embedding_model_flag == False:
			return "No VectorDB Loaded"
		else:
			self.vectorstore = FAISS.load_local(vectordb_name, self.embedding_model)
		
	def create_open_source_embedding_model(self,embedding_model_name):
		self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,model_kwargs={'device':   'cuda'})
		self.embedding_model_flag = True
		
	def load_vectordb(self,vectordb_path, embedding_model):
		if self.embedding_model_flag == False:
			return "No VectorDB Loaded"
		else:
			self.vectorstore = FAISS.load_local(vectordb_path,self.embedding_model)
			
	def create_huggingface_qa_chain(self):
		if self.vectorstore == None:
			return 'NO VectorDB Loaded'
		else:
			llm = HuggingFacePipeline(pipeline=self.pipe, model_kwargs={'temperature':0})
			chain_type_kwargs = self._get_prompt_for_huggingface_llama_docqa()
			self.qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                       chain_type = "stuff",
                                       retriever = self.vectorstore.as_retriever(),
                                       chain_type_kwargs = chain_type_kwargs,
                                       return_source_documents = True)
			self.qa_chain_flag = True

	def _get_prompt_for_huggingface_llama_docqa(self):
		B_INST, E_INST = "[INST]", "[/INST]"
		B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
		sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

		instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""

		SYSTEM_PROMPT = B_SYS + sys_prompt + E_SYS
		prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST

		llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

## Create chain type argument for the new prompt
		chain_type_kwargs = {"prompt": llama_prompt}

		return chain_type_kwargs
	
	def query_qa_chain(self, user_input):
		if self.qa_chain_flag == False:
			print('LOAD A QA CHAIN')
		else:
			return self.qa_chain(user_input)

	def _chunk_and_apply_prompt(self,text, max_length = 5000):
		directory = 'temp_summaries'
		if os.path.isdir(directory) != True:
			os.mkdir(directory)
		start = 0
		for i in tqdm.tqdm(range(max_length,len(text),max_length)):
			temp_transcript = text[start:max_length]
			output = self.execute_prompt(temp_transcript)
			with open(f'{directory}/{i}.txt','w') as f:
				f.write(output)
			start += i
			
	def _combine_temp_chunks(self,delete = True):
		prefix = 'temp_summaries'
		files = os.listdir(prefix)
		text = []
		for file in files:
			file = f'{prefix}/{file}'
			if '.txt' in file:
				with open(file,'r') as f:
					text.append(f.read())
				if delete == True:
					os.remove(file)
				else:
					pass
		text = ' '.join(text)
		return text
	
	def large_document_prompt_execution(self, text, max_length = 5000, delete = True):
		self._chunk_and_apply_prompt(text, max_length = max_length)
		return self._combine_temp_chunks(delete = delete)
	
	def load_open_ai_endpoint_params(self,api_key,deployment_model,api_version,api_base):

		self.api_url = f"{api_base}/openai/deployments/{deployment_model}/completions?api-version={api_version}"
		self.openAI_api_key = api_key
		self.headers =  {"api-key": self.openAI_api_key}
		self.open_ai_model_flag = True

	def execute_open_ai_query(self,context, temperature = 0, max_tokens = 5000):

		if self.open_ai_model_flag == False:
			return "Load in an openAI model with teh load_open_ai method"
		
		prompt = self._create_prompt(context)

		json_data = json_data = {
 							 "prompt": prompt,
  							 "temperature":temperature,
  							"max_tokens": max_tokens			
										}

		response = requests.post(self.api_url, json=json_data, headers=self.headers)
		response = self._unpack_response_json(response)

		return response

	def _unpack_response_json(self,response):
		try:
			response = response.json()
			response = response['choices']
			outs = []
			for item in response:
				out = item['text'].replace('\n','')
				outs.append(out)
			
		
			return outs
		except Exception as e:
			print(e)
			return None
			
	def execute_open_ai_query_large_doc(self, context,character_split = 10000, overlap = 1000,
									  temperature = 0, max_tokens = 3000):
		
		responses = []
		start = 0


		for i in range(character_split,len(context),character_split):
			if start == 0:
				temp_context = context[start:i]

			else:
				temp_context = context[start-overlap : i]

			prompt = self._create_prompt(temp_context)
			json_data = {"prompt": prompt,
  						"temperature":temperature,
  						"max_tokens": max_tokens}
			
			response = requests.post(self.api_url, json=json_data, headers=self.headers)
			response = self._unpack_response_json(response)
			try:
				responses = responses + response
			except Exception as e:
				print(e)
				pass
			start = i
		return responses
