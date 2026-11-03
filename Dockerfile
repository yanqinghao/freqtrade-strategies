FROM freqtradeorg/freqtrade:2026.10

# Keep upstream execution and RPC unchanged; extend Telegram commands only.
RUN cp /freqtrade/freqtrade/rpc/telegram.py /freqtrade/freqtrade/rpc/_telegram_base.py
COPY telegram.py /freqtrade/freqtrade/rpc/telegram.py
