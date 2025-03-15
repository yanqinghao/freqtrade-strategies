FROM freqtradeorg/freqtrade:stable

RUN pip install langchain langchain_openai matplotlib mplfinance dotenv psycopg2-binary

COPY telegram.py /freqtrade/freqtrade/rpc/telegram.py
RUN mkdir -p /freqtrade/freqtrade/freqllm
COPY analysis_agent.py /freqtrade/freqtrade/freqllm/analysis_agent.py
COPY key_level_agent.py /freqtrade/freqtrade/freqllm/key_level_agent.py
COPY db_manager.py /freqtrade/freqtrade/freqllm/db_manager.py
