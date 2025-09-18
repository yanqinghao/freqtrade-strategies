# user_data/plugins/html_sanitizer.py
import re

# Telegram HTML 白名单
ALLOWED = (
    'a',
    'b',
    'strong',
    'i',
    'em',
    'u',
    's',
    'strike',
    'del',
    'code',
    'pre',
    'blockquote',
    'tg-spoiler',
    'span',
)

# 允许的起始/结束标签匹配（尽量宽容，但仅限白名单）
ALLOWED_OPEN = re.compile(rf"<((?:{'|'.join(ALLOWED)})(?:\s+[^<>]*?)?)>", re.I)
ALLOWED_CLOSE = re.compile(rf"</((?:{'|'.join(ALLOWED)}))\s*>", re.I)

# span 仅允许 class="tg-spoiler"
SPAN_CLASS_FIX = re.compile(r'<span(?![^>]*\bclass\s*=\s*"(?:tg-spoiler)")([^>]*)>', re.I)

# <a> 仅允许 href
A_TAG = re.compile(r'<a([^>]*)>', re.I)


def _clean_a(m: re.Match) -> str:
    attrs = m.group(1) or ''
    mhref = re.search(r'href\s*=\s*"([^"]+)"', attrs, re.I)
    return f'<a href="{mhref.group(1)}">' if mhref else '<a>'


def _escape_non_allowed_angles(text: str) -> str:
    """
    仅对白名单标签之外的 < 和 > 进行转义：
      - 保留 ALLOWED_OPEN / ALLOWED_CLOSE 不变
      - 其它所有 < / > 替换成 &lt; / &gt;
    实现思路：先把允许的标签替换为占位符，再全局转义，最后还原占位符。
    """
    tokens = []

    def keep_open(m: re.Match):
        tokens.append(m.group(0))
        return f"\uFFF0{len(tokens) - 1}\uFFF0"  # 占位符: U+FFF0 包裹索引

    def keep_close(m: re.Match):
        tokens.append(m.group(0))
        return f"\uFFF0{len(tokens) - 1}\uFFF0"

    tmp = ALLOWED_OPEN.sub(keep_open, text)
    tmp = ALLOWED_CLOSE.sub(keep_close, tmp)

    # 只转义不是实体开头的 & ： &amp; / &#123; / &#x1F4A9; 都不再二次转义
    tmp = re.sub(r'&(?![a-zA-Z]+;|#\d+;|#x[0-9A-Fa-f]+;)', '&amp;', tmp)
    # 然后再安全地转义尖括号
    tmp = tmp.replace('<', '&lt;').replace('>', '&gt;')

    # 还原允许的标签
    def restore_token(m: re.Match) -> str:
        idx = int(m.group(1))
        return tokens[idx]

    tmp = re.sub(r'\uFFF0(\d+)\uFFF0', restore_token, tmp)
    return tmp


def sanitize_telegram_html(raw: str) -> str:
    if not raw:
        return ''

    html = raw

    # 清理 <span> 属性，仅保留 class="tg-spoiler"
    html = SPAN_CLASS_FIX.sub('<span>', html)
    # 清理 <a>，只保留 href
    html = A_TAG.sub(_clean_a, html)

    # 将所有非白名单标签的尖括号转义
    html = _escape_non_allowed_angles(html)

    # 这里不做任何“补闭合”的魔改，仅做极轻微纠偏：若 <pre>/<code>/<blockquote> 开闭数量不等，进行配平
    for tag in ('pre', 'code', 'blockquote'):
        opens = len(re.findall(rf"<{tag}(?:\s+[^<>]*?)?>", html, re.I))
        closes = len(re.findall(rf"</{tag}\s*>", html, re.I))
        if opens > closes:
            html += f"</{tag}>" * (opens - closes)
        elif closes > opens:
            # 删掉多余的闭合，避免越权修复
            html = re.sub(rf"</{tag}\s*>", '', html, count=(closes - opens), flags=re.I)

    # 压缩多余空行
    html = re.sub(r'\n{3,}', '\n\n', html).strip()
    return html
