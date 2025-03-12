FROM freqtradeorg/freqtrade:stable

RUN pip install langchain langchain_openai matplotlib mplfinance

COPY telegram.py /freqtrade/freqtrade/rpc/telegram.py
RUN mkdir -p /freqtrade/freqtrade/freqllm
COPY analysis_agent.py /freqtrade/freqtrade/freqllm/analysis_agent.py
