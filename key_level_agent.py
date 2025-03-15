import os
import json
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field, validator


class TradingSignal(BaseModel):
    """Trading signal with entry and exit points."""

    direction: str = Field(description="Trading direction: 'long' or 'short'")
    entry_points: List[float] = Field(description='List of entry price points, sorted')
    exit_points: List[float] = Field(description='List of target/exit price points, sorted')
    stop_loss: float = Field(description='Stop loss price level')
    risk_reward: Optional[float] = Field(None, description='Risk-reward ratio if available')

    @validator('direction')
    def validate_direction(cls, v):
        if v not in ['long', 'short']:
            raise ValueError(f"Direction must be 'long' or 'short', got {v}")
        return v

    @validator('entry_points')
    def validate_entry_points(cls, v):
        if not v:
            raise ValueError('Entry points list cannot be empty')
        return sorted(v)

    @validator('exit_points')
    def validate_exit_points(cls, v, values):
        if not v:
            raise ValueError('Exit points list cannot be empty')

        # Sort based on direction
        direction = values.get('direction')
        if direction == 'long':
            return sorted(v)  # For long positions, targets are ascending
        else:
            return sorted(v, reverse=True)  # For short positions, targets are descending


class TradingSignalExtractor:
    """
    Simple agent to extract trading signals from analysis text and convert to JSON
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gpt-3.5-turbo'):
        """
        Initialize the Trading Signal Extractor

        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
            model_name: Name of the language model to use
        """
        # Use provided API key or get from environment
        llm_api_key = os.environ['LLM_API_KEY']
        llm_base_url = os.environ['LLM_BASE_URL']
        llm_model_name = os.environ['LLM_MODEL_NAME']
        self.llm = ChatOpenAI(
            temperature=0.1, model_name=llm_model_name, base_url=llm_base_url, api_key=llm_api_key
        )

        # Define the extraction prompt template
        self.extraction_template = """
        You are a trading signal extraction specialist. Extract the specific trading signals from the following analysis text:

        ```
        {text}
        ```

        Output ONLY a JSON array in the following format:
        [
          {{
            "direction": "long" or "short",
            "entry_points": [array of entry price numbers, sorted from low to high for long, high to low for short],
            "exit_points": [array of target/exit price numbers, sorted from low to high for long, high to low for short],
            "stop_loss": stop loss price number,
            "risk_reward": risk/reward ratio number (optional)
          }}
        ]

        Rules:
        1. If both long and short signals are present, include them as separate objects in the array
        2. Entry points should include all mentioned entry prices as numbers
        3. Exit points should include all target prices as numbers
        4. For long positions: sort entry_points and exit_points from lowest to highest
        5. For short positions: sort entry_points and exit_points from highest to lowest
        6. Include risk_reward only if it's explicitly mentioned

        Parse all numbers as floating point values, not strings. Respond with ONLY the JSON array.
        """

        self.prompt = PromptTemplate(input_variables=['text'], template=self.extraction_template)

        # Create the extraction chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def extract_signals(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract trading signals from analysis text

        Args:
            text: Technical analysis text

        Returns:
            List of dictionaries containing trading signals
        """
        try:
            # Process the text through the LLM chain
            response = self.chain.run(text=text)
            # Extract the JSON string from the response
            json_str = self._extract_json_from_response(response)

            # Parse the JSON
            if json_str:
                try:
                    parsed_data = json.loads(json_str)

                    # Handle both single-object and list responses
                    if isinstance(parsed_data, dict):
                        signals = [parsed_data]
                    elif isinstance(parsed_data, list):
                        signals = parsed_data
                    else:
                        raise ValueError(f"Unexpected JSON structure: {type(parsed_data)}")

                    # Validate each signal
                    valid_signals = []
                    for signal in signals:
                        try:
                            validated = TradingSignal(**signal)
                            valid_signals.append(validated.dict())
                        except Exception as e:
                            print(f"Validation error for signal: {e}")

                    return valid_signals
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    # Fall back to regex extraction if JSON parsing fails
                    return self._fallback_extract(text)
            else:
                # Fall back to regex extraction if no JSON found
                return self._fallback_extract(text)

        except Exception as e:
            print(f"Error extracting signals: {e}")
            return self._fallback_extract(text)

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON string from LLM response

        Args:
            response: LLM response string

        Returns:
            Cleaned JSON string or empty string if not found
        """
        # Look for JSON structure with or without code blocks
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)

        if json_match:
            # JSON was in a code block
            return json_match.group(1).strip()

        # Try to find JSON without code blocks
        # Look for brackets enclosing the entire response
        if response.strip().startswith('[') and response.strip().endswith(']'):
            return response.strip()

        # Try to find any JSON array in the response
        json_match = re.search(r'(\[.*\])', response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        return ''

    def _fallback_extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback method to extract trading signals using regex patterns

        Args:
            text: Technical analysis text

        Returns:
            List of dictionaries containing trading signals
        """
        results = []

        # Determine if long signal is present
        long_present = bool(re.search(r'做多|bullish|long|buy|买入|买|看涨', text.lower()))

        # Determine if short signal is present
        short_present = bool(re.search(r'做空|bearish|short|sell|卖出|卖|看跌', text.lower()))

        # Extract direction(s)
        directions = []
        if long_present and short_present:
            # Check if one is more emphasized
            long_emphasis = len(re.findall(r'做多|bullish|long|buy|买入|买|看涨', text.lower()))
            short_emphasis = len(re.findall(r'做空|bearish|short|sell|卖出|卖|看跌', text.lower()))

            if long_emphasis > short_emphasis * 1.5:
                directions = ['long']
            elif short_emphasis > long_emphasis * 1.5:
                directions = ['short']
            else:
                directions = ['long', 'short']
        elif long_present:
            directions = ['long']
        elif short_present:
            directions = ['short']
        else:
            # Default to long if direction is unclear
            directions = ['long']

        # Process each direction
        for direction in directions:
            # Initialize signal structure
            signal = {
                'direction': direction,
                'entry_points': [],
                'exit_points': [],
                'stop_loss': None,
            }

            # Extract entry points
            entry_patterns = [
                r'入场.*?(\d+\.\d+)',
                r'入场价格.*?(\d+\.\d+)',
                r'entry.*?(\d+\.\d+)',
                r'入场.*?(\d+\.\d+[–-]\d+\.\d+)',
            ]

            for pattern in entry_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if '–' in match or '-' in match:
                        # Handle price ranges
                        parts = re.split(r'[–-]', match)
                        for part in parts:
                            try:
                                signal['entry_points'].append(float(part.strip()))
                            except ValueError:
                                continue
                    else:
                        try:
                            signal['entry_points'].append(float(match.strip()))
                        except ValueError:
                            continue

            # Extract exit points
            exit_patterns = [
                r'目标.*?(\d+\.\d+)',
                r'target.*?(\d+\.\d+)',
                r'止盈.*?(\d+\.\d+)',
                r'tp.*?(\d+\.\d+)',
                r'profit.*?(\d+\.\d+)',
            ]

            for pattern in exit_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        signal['exit_points'].append(float(match.strip()))
                    except ValueError:
                        continue

            # Extract stop loss
            stop_loss_patterns = [r'止损.*?(\d+\.\d+)', r'stop.*?(\d+\.\d+)', r'sl.*?(\d+\.\d+)']

            for pattern in stop_loss_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        signal['stop_loss'] = float(match.group(1).strip())
                        break
                    except ValueError:
                        continue

            # Extract risk/reward ratio
            rr_patterns = [
                r'风险收益比.*?(\d+(?:\.\d+)?)',
                r'risk[/\-]reward.*?(\d+(?:\.\d+)?)',
                r'r:r.*?(\d+(?:\.\d+)?)',
                r'rr.*?(\d+(?:\.\d+)?)',
            ]

            for pattern in rr_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        signal['risk_reward'] = float(match.group(1).strip())
                        break
                    except ValueError:
                        continue

            # Ensure we have required fields with valid values
            if signal['entry_points'] and signal['exit_points']:
                # Sort appropriately
                if direction == 'long':
                    signal['entry_points'] = sorted(signal['entry_points'])
                    signal['exit_points'] = sorted(signal['exit_points'])
                else:
                    signal['entry_points'] = sorted(signal['entry_points'], reverse=True)
                    signal['exit_points'] = sorted(signal['exit_points'], reverse=True)

                # Generate stop loss if not found
                if signal['stop_loss'] is None and signal['entry_points']:
                    if direction == 'long':
                        signal['stop_loss'] = min(signal['entry_points']) * 0.95
                    else:
                        signal['stop_loss'] = max(signal['entry_points']) * 1.05

                results.append(signal)

        return results

    def consolidate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate multiple signals of the same direction into optimized signals

        Args:
            signals: List of extracted trading signals

        Returns:
            Consolidated list of trading signals
        """
        if not signals:
            return []

        # Group signals by direction
        long_signals = [s for s in signals if s['direction'] == 'long']
        short_signals = [s for s in signals if s['direction'] == 'short']

        consolidated_signals = []

        # Consolidate long signals
        if long_signals:
            # For long positions, use lowest entry and lowest exit
            all_entries = [entry for signal in long_signals for entry in signal['entry_points']]
            all_exits = [
                exit_point for signal in long_signals for exit_point in signal['exit_points']
            ]
            all_stops = [
                signal['stop_loss'] for signal in long_signals if signal['stop_loss'] is not None
            ]

            if all_entries and all_exits and all_stops:
                best_long = {
                    'direction': 'long',
                    'entry_points': [min(all_entries)],  # Lowest entry
                    'exit_points': [min(all_exits)],  # Lowest exit (first target)
                    'stop_loss': min(all_stops),  # Most conservative stop
                    'risk_reward': None,
                }

                # Calculate risk/reward if possible
                entry = best_long['entry_points'][0]
                exit_point = best_long['exit_points'][0]
                stop = best_long['stop_loss']

                if entry and stop and exit_point:
                    risk = abs(entry - stop)
                    reward = abs(exit_point - entry)
                    if risk > 0:
                        best_long['risk_reward'] = round(reward / risk, 2)

                consolidated_signals.append(best_long)

        # Consolidate short signals
        if short_signals:
            # For short positions, use highest entry and highest exit (lowest price)
            all_entries = [entry for signal in short_signals for entry in signal['entry_points']]
            all_exits = [
                exit_point for signal in short_signals for exit_point in signal['exit_points']
            ]
            all_stops = [
                signal['stop_loss'] for signal in short_signals if signal['stop_loss'] is not None
            ]

            if all_entries and all_exits and all_stops:
                best_short = {
                    'direction': 'short',
                    'entry_points': [max(all_entries)],  # Highest entry
                    'exit_points': [max(all_exits)],  # Highest exit (first target)
                    'stop_loss': max(all_stops),  # Most conservative stop
                    'risk_reward': None,
                }

                # Calculate risk/reward if possible
                entry = best_short['entry_points'][0]
                exit_point = best_short['exit_points'][0]
                stop = best_short['stop_loss']

                if entry and stop and exit_point:
                    risk = abs(entry - stop)
                    reward = abs(exit_point - entry)
                    if risk > 0:
                        best_short['risk_reward'] = round(reward / risk, 2)

                consolidated_signals.append(best_short)

        return consolidated_signals

    def process_analysis(
        self,
        text: str,
        save_to_file: bool = False,
        file_path: Optional[str] = None,
        consolidate: bool = False,
    ) -> Dict[str, Any]:
        """
        Process analysis text and return JSON results

        Args:
            text: Technical analysis text
            save_to_file: Whether to save the results to a JSON file
            file_path: Path to save the JSON file (if None, generates a timestamp-based name)
            consolidate: Whether to consolidate multiple signals of the same direction

        Returns:
            Dictionary with extraction results
        """
        # Extract signals
        signals = self.extract_signals(text)

        # Consolidate signals if requested
        if consolidate:
            signals = self.consolidate_signals(signals)

        # Create results dictionary
        results = {'signals': signals, 'count': len(signals), 'timestamp': self._get_timestamp()}

        # Save to file if requested
        if save_to_file:
            self._save_to_file(results, file_path)

        return results

    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime

        return datetime.now().isoformat()

    def _save_to_file(self, data: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """
        Save data to JSON file

        Args:
            data: Data to save
            file_path: Path to save file (generates one if None)

        Returns:
            Path to saved file
        """
        if file_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('trading_signals', exist_ok=True)
            file_path = f"trading_signals/signals_{timestamp}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return file_path

    def extract_to_json_string(self, text: str, pretty: bool = True, consolidate: bool = False):
        """
        Extract trading signals and return as JSON string

        Args:
            text: Technical analysis text
            pretty: Whether to format the JSON with indentation
            consolidate: Whether to consolidate multiple signals of the same direction

        Returns:
            JSON string of extracted signals
        """
        raw_signals = self.extract_signals(text)

        signals = None

        # Consolidate signals if requested
        if consolidate:
            signals = self.consolidate_signals(raw_signals)

        return raw_signals, signals


# Example usage
if __name__ == '__main__':
    # Example analysis text
    analysis_text = """
    📌 1. 币种背景
    WLD/USDT 于 2023-07-24 上线，截至 2025-03-12，已上线约 1.5 年。该币种仍处于发展阶段，市场成熟度中等，流动性逐步增强（日成交量在 1.8 亿至 3.7 亿 USDT 之间），但长期价格波动较大（从 1.01 高点跌至 0.70 低点）。近期价格处于历史低位（0.70–0.83），可能吸引短期投机资金，但尚未形成明确长期趋势。
    📌 2. 多时间周期分析
    📌 # 1D（日线）
    • 趋势：看跌（ADX=43.57，价格低于 SMA20/SMA50，MA 未交叉）。
    • 动能：RSI（33）超卖但未反转，MACD 负值且未金叉，短期反弹可能为技术性修复。
    • 波动性：价格位于布林带中轨下方，但已脱离下轨，下行空间收窄。
    • 冲突点：超卖信号（RSI、Stoch）与价格下行趋势并存，需警惕反弹风险。
    📌 # 4H（4 小时线）
    • 趋势：中性（ADX=43.08，价格突破 SMA20 但未站稳）。
    • 动能：RSI（46.6）中性偏强，MACD 金叉且柱状图转正，短期反弹信号。
    • 关键阻力：0.83（近期高点），突破可能触发空头回补。
    📌 # 1H（小时线）
    • 趋势：看涨（价格高于 SMA20/SMA50，ADX=16.52 显示趋势较弱）。
    • 动能：RSI（59）、Stoch（76/57）、CCI（141）显示超买，但 MACD 金叉延续。
    • 波动性：价格触及布林带上轨（0.8295），需关注回调风险。
    📌 # 15m（15 分钟线）
    • 趋势：中性（ADX=24.93，价格短暂突破布林带上轨后回落）。
    • 动能：RSI（72）、Stoch（92/91）严重超买，CCI（139）背离，短期回调概率高。
    信号一致性：
    • 长期（1D）看跌 vs 短期（1H/4H）反弹：日线主导趋势，但短期存在超跌反弹动能。
    • 超买冲突：1H/15m 显示超买，但 4H 动能尚未耗尽，需关注关键阻力位突破情况。
    📌 3. 交易机会评估
    • 做多机会：短期（1H/4H）反弹信号明确，但需突破 0.83 阻力确认趋势反转。
    • 做空机会：若价格在 0.83 附近受阻，结合 15m 超买信号可轻仓试空。
    • 最强信号：1H 周期（RSI/CCI/MACD 共振看涨），但需警惕日线趋势压制。
    📌 4. 入场建议
    📌 # 做多策略（短线）
    • 入场时机：
    • 1H：价格回调至 0.80（布林带中轨/SMA20）或突破 0.83 后回踩确认。
    • 4H：MACD 柱状图持续扩大且站稳 SMA20（0.8098）。
    • 价格水平：0.80–0.81（支撑区），0.83（突破追多）。
    📌 # 做空策略（保守）
    • 入场时机：价格在 0.83 附近出现 15m 看跌信号（如长上影线、RSI 顶背离）。
    • 价格水平：0.83–0.84（阻力区）。
    📌 5. 风险管理
    • 止损位：
    • 做多：0.77（日线前低下方）。
    • 做空：0.85（突破近期高点）。
    • 止盈目标：
    • 做多：0.88（4H 前高）、0.95（日线中轨）。
    • 做空：0.75（4H 支撑）、0.70（心理关口）。
    📌 6. 信号冲突解释
    • 日线 vs 小时线：日线长期超卖未反转，但短期资金涌入推升价格，形成技术性反弹。
    • 1H 超买 vs 4H 中性：1H 超买可能引发回调，但 4H 动能尚未耗尽，需观察价格能否站稳关键位。
    📌 总结与交易建议
    当前观点：短期反弹动能存在，但日线趋势压制明显，建议以短线做多为主，严格止损。
    最终建议：
    • 做多：在 0.80–0.81 区间轻仓入场，止损 0.77，目标 0.88。
    • 激进追多：若放量突破 0.83，回踩 0.82 加仓，止损 0.79。
    • 做空：仅在 0.83 附近出现明确反转信号后试空，止损 0.85。
    加粗结论：短线看涨，但需严格止损；日线趋势未反转前，避免重仓持有。
    """

    # Initialize extractor
    extractor = TradingSignalExtractor()

    # Extract signals and print as JSON
    # print("Original signals:")
    # json_output = extractor.extract_to_json_string(analysis_text)
    # print(json_output)

    # Extract signals, consolidate, and print as JSON
    print('\nConsolidated signals:')
    json_output = extractor.extract_to_json_string(analysis_text, consolidate=True)
    print(json_output)
