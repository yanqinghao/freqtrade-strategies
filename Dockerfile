FROM freqtradeorg/freqtrade:stable

RUN pip install langchain langchain_openai matplotlib mplfinance dotenv psycopg2-binary fastmcp openai

COPY telegram.py /freqtrade/freqtrade/rpc/telegram.py
COPY rpc.py /freqtrade/freqtrade/rpc/rpc.py
COPY freqtradebot.py /freqtrade/freqtrade/freqtradebot.py
RUN mkdir -p /freqtrade/freqtrade/freqllm
COPY analysis_agent.py /freqtrade/freqtrade/freqllm/analysis_agent.py
COPY ai_agent.py /freqtrade/freqtrade/freqllm/ai_agent.py
COPY pagination_utils.py /freqtrade/freqtrade/freqllm/pagination_utils.py
COPY html_sanitizer.py /freqtrade/freqtrade/freqllm/html_sanitizer.py
COPY key_level_agent.py /freqtrade/freqtrade/freqllm/key_level_agent.py
COPY db_manager.py /freqtrade/freqtrade/freqllm/db_manager.py
