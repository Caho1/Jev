"""Frozen, hand-authored DOM diagnostics. Gold never enters the model request."""

from html import escape
import random


def document(body):
    return ('<!doctype html><meta charset="utf-8"><title>Browser task fixture</title>'
            '<style>body{font:18px Arial;padding:24px}button,input,select,a{margin:10px;padding:8px}'
            'label{display:inline-block;margin:8px}section{padding:12px}</style>' + body)


def cases():
    rows = []

    def add(name, body, en, zh, operation, target=None, value=None, scroll=None, history=None):
        for language, goal in [('en', en), ('zh', zh)]:
            rows.append(dict(id=f'{name}-{language}', family=name, language=language,
                             html=document(body), goal=goal, operation=operation,
                             target=target, value=value, scroll=scroll, history=history or []))

    def buttons(labels, seed):
        items = [(f'b{i}', text) for i, text in enumerate(labels)]
        random.Random(seed).shuffle(items)
        return ''.join(f'<button id="{key}">{escape(text)}</button>' for key, text in items)

    add('click-button', '<h1>Workspace</h1>'+buttons(['Settings', 'Help', 'Profile', 'Projects'], 7),
        'Open Settings.', '打开 Settings 设置。', 'CLICK', 'b0')
    add('click-link', '<h1>Reading room</h1><a id="a" href="#latency">Where the milliseconds go</a>'
        '<a href="#confidence">Confidence is not correctness</a><a href="#choices">A browser is a choice</a>',
        'Open the article Where the milliseconds go.', '打开文章 Where the milliseconds go。', 'CLICK', 'a')
    add('click-checkbox', '<h1>Stay filters</h1><label><input type="checkbox" id="a">Free cancellation</label>'
        '<label><input type="checkbox" checked>Breakfast included</label><button>Reset filters</button>',
        'Enable Free cancellation. Keep Breakfast included enabled.',
        '勾选 Free cancellation，保持 Breakfast included 已勾选。', 'CLICK', 'a')
    add('click-radio', '<h1>Flight type</h1><label><input type="radio" name="trip" checked>Round trip</label>'
        '<label><input type="radio" name="trip" id="a">One way</label><button>Help</button>',
        'Switch the flight type to One way.', '把航班类型改成 One way 单程。', 'CLICK', 'a')
    add('click-submit', '<h1>Find stays</h1><input aria-label="Destination" value="Lisbon">'
        '<button id="a">Search</button><button>Clear</button><p>The search has not been applied.</p>',
        'Search for stays in Lisbon. The destination is already filled; apply the search.',
        '搜索 Lisbon 住宿，目的地已填好，请执行搜索。', 'CLICK', 'a')
    add('click-autocomplete', '<h1>Flight search</h1><input role="combobox" aria-label="Destination" value="London">'
        '<button role="option">London Heathrow Airport (LHR)</button>'
        '<button role="option" id="a">London Gatwick Airport (LGW)</button>',
        'Choose London Gatwick Airport (LGW) from the destination suggestions.',
        '从目的地建议中选择 London Gatwick Airport (LGW)。', 'CLICK', 'a')
    for i, (label, other, term, zh) in enumerate([
        ('Destination', 'Guest name', 'Lisbon', '在 Destination 目的地框填写 Lisbon。'),
        ('Search books', 'Newsletter email', 'Dune', '在 Search books 框搜索 Dune，先输入书名。'),
        ('First name', 'Last name', 'Ada', '在 First name 框填写 Ada，Last name 保持不变。'),
        ('Email', 'City', 'ada@example.test', '把 Email 改为 ada@example.test，City 保持不变。'),
    ]):
        values = ['', '', '', 'old@example.test']
        add(f'type-{i}', f'<h1>Form</h1><label>{other}<input aria-label="{other}" value="existing"></label>'
            f'<label>{label}<input id="a" aria-label="{label}" value="{values[i]}"></label><button>Submit</button>',
            f'Enter {term} in {label}. Leave {other} unchanged.', zh, 'TYPE_TEXT', 'a')
    for i, (label, options, wanted, zh) in enumerate([
        ('Currency', ['USD', 'EUR', 'GBP'], 'EUR', '把 Currency 货币切换为 EUR。'),
        ('Language', ['English', 'Chinese', 'French'], 'Chinese', '把 Language 切换为 Chinese。'),
        ('Sort results', ['Recommended', 'Price low to high', 'Newest'], 'Price low to high', '把结果排序设为 Price low to high。'),
        ('Stay category', ['All stays', 'Design', 'Nature'], 'Design', '把 Stay category 设为 Design。'),
    ]):
        option_html = ''.join(f'<option value="v{j}">{escape(o)}</option>' for j, o in enumerate(options))
        add(f'select-{i}', f'<h1>Preferences</h1><label>{label}<select id="a">{option_html}</select></label>'
            '<select aria-label="Items per page"><option>10</option><option>20</option></select><button>Help</button>',
            f'Set {label} to {wanted}.', zh, 'SELECT', 'a', f'v{options.index(wanted)}')
    add('done-article', '<h1>Where the milliseconds go</h1><p>Measuring latency in browser agents.</p><button>Back</button>',
        'Open the article Where the milliseconds go. Stop once its content is visible.',
        '打开 Where the milliseconds go 文章，内容可见后停止。', 'DONE')
    add('done-toggle', '<h1>Stay filters</h1><label><input type="checkbox" checked>Free cancellation</label><button>Reset</button>',
        'Enable Free cancellation. Stop once it is enabled.', '启用 Free cancellation，已勾选后停止。', 'DONE')
    add('done-select', '<h1>Preferences</h1><select aria-label="Currency"><option>USD</option><option selected>EUR</option></select>',
        'Set Currency to EUR. Stop once selected.', '把 Currency 设为 EUR，已选中后停止。', 'DONE')
    add('done-search', '<h1>Search results for Lisbon</h1><p>Search applied. 1 stay found in Lisbon: Casa Flora.</p>'
        '<input aria-label="Destination" value="Lisbon"><button>Search</button>',
        'Search for stays in Lisbon. Stop when the search results are visible; do not open a stay.',
        '搜索 Lisbon 住宿，搜索结果可见后停止，不要打开住宿。', 'DONE')
    add('wait-results', '<h1>Search submitted</h1><p>Loading results for Lisbon… Please wait.</p><button disabled>Search</button>',
        'Find stays in Lisbon. The search was submitted; wait for results to finish loading.',
        '查找 Lisbon 住宿，搜索已提交，等待结果加载完成。', 'WAIT')
    add('wait-control', '<h1>Preparing download</h1><p>Your report is being generated. Download will become available shortly.</p>'
        '<button disabled>Download report</button>',
        'Download the report when generation finishes.', '报告生成完成后下载报告。', 'WAIT')
    add('blocked-upload', '<h1>Attach document</h1><p>To continue, upload a PDF using the file picker.</p><input type="file">',
        'Upload my PDF. If none of the supported operations can upload files, report blocked.',
        '上传我的 PDF；如果支持的操作都无法上传文件，报告受阻。', 'BLOCKED')
    add('blocked-canvas', '<h1>Canvas diagram editor</h1><canvas width="400" height="200"></canvas>'
        '<p>The diagram can only be edited by drawing on the canvas; no accessible controls are available.</p>',
        'Draw a circle in the canvas. Report blocked if no supported operation can do it.',
        '在画布中画一个圆，若没有支持的操作可完成则报告受阻。', 'BLOCKED')
    add('scroll-down', '<h1>Long article</h1><p>Scroll to the bottom to see the references.</p>'
        '<div style="height:1500px">Article content</div><h2>References</h2><a href="#source">Source</a>',
        'Scroll down to view the references at the bottom of the article.', '向下滚动查看文章底部的参考资料。', 'SCROLL_DOWN')
    add('scroll-up', '<h1>Article overview</h1><div style="height:1500px">Article body</div><p>End of article</p>',
        'Scroll up to view the article overview at the top.', '向上滚动查看文章顶部的概述。', 'SCROLL_UP', scroll=1600)
    return rows


def closed_cases():
    rows = []
    scenarios = [
        ('article', '<h1>Reading room</h1><button onclick="document.querySelector(\'main\').innerHTML=\'<h1>Where the milliseconds go</h1><p>Article content: latency measurements.</p>\'">Where the milliseconds go</button>'
         '<button>Confidence is not correctness</button>',
         'Open the article Where the milliseconds go.', '打开文章 Where the milliseconds go。',
         'document.querySelector("main h1")?.textContent === "Where the milliseconds go"'),
        ('toggle-save', '<h1>Preferences</h1><label><input id="free" type="checkbox">Free cancellation</label>'
         '<label><input id="breakfast" type="checkbox" checked>Breakfast included</label>'
         '<button onclick="document.querySelector(\'output\').textContent=document.querySelector(\'#free\').checked && document.querySelector(\'#breakfast\').checked ? \'Preferences saved: Free cancellation and Breakfast included enabled\' : \'Incorrect preferences saved\'">Save preferences</button><output></output>',
         'Enable Free cancellation, keep Breakfast included enabled, then save the preferences.',
         '启用 Free cancellation，保持 Breakfast included 启用，然后保存偏好。',
         'document.querySelector("output")?.textContent === "Preferences saved: Free cancellation and Breakfast included enabled" && document.querySelector("#free").checked && document.querySelector("#breakfast").checked'),
        ('select-apply', '<h1>Currency preferences</h1><select id="currency" aria-label="Currency"><option>USD</option><option>EUR</option><option>GBP</option></select>'
         '<button onclick="document.querySelector(\'output\').textContent=\'Applied currency: \'+document.querySelector(\'#currency\').value">Apply currency</button><output></output>',
         'Set Currency to EUR and apply the currency preference.', '把 Currency 设为 EUR 并应用货币偏好。',
         'document.querySelector("#currency")?.value === "EUR" && document.querySelector("output")?.textContent === "Applied currency: EUR"'),
        ('two-clicks', '<h1>Home</h1><button onclick="document.querySelector(\'main\').innerHTML=\'<h1>Help center</h1><button onclick=&quot;document.querySelector(\\\'main\\\').innerHTML=\\\'<h1>Contact support</h1><p>Support contact details</p>\\\'&quot;>Contact support</button><button>FAQ</button>\'">Help center</button><button>Settings</button>',
         'Open Help center, then open Contact support.', '打开 Help center，然后打开 Contact support。',
         'document.querySelector("main h1")?.textContent === "Contact support"'),
    ]
    for name, body, en, zh, verify in scenarios:
        for lang, goal in [('en', en), ('zh', zh)]:
            rows.append(dict(id=f'{name}-{lang}', language=lang, html=document('<main>'+body+'</main>'),
                             goal=goal, verify=verify))
    return rows
