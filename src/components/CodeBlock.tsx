import { Highlight, type PrismTheme } from 'prism-react-renderer'

// 瑞士蓝配色的语法高亮主题：浅底、IKB 蓝关键字、低饱和注释。
const swissTheme: PrismTheme = {
  plain: { color: '#1b1f24', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: '#9aa1a9', fontStyle: 'italic' } },
    { types: ['keyword', 'builtin', 'boolean'], style: { color: '#002fa7', fontWeight: 'bold' } },
    { types: ['function', 'class-name'], style: { color: '#7a1fa2' } },
    { types: ['string', 'char', 'attr-value'], style: { color: '#0a7d52' } },
    { types: ['number', 'constant', 'symbol'], style: { color: '#b23a00' } },
    { types: ['operator', 'punctuation'], style: { color: '#5b6168' } },
    { types: ['property', 'tag', 'attr-name'], style: { color: '#0050b3' } },
    { types: ['variable', 'parameter'], style: { color: '#1b1f24' } },
  ],
}

export function CodeBlock({ code, language = 'tsx', title }: {
  code: string
  language?: string
  title?: string
}) {
  return (
    <div className="codeblock">
      {title && <div className="codeblock-title">{title}</div>}
      <Highlight theme={swissTheme} code={code.trim()} language={language}>
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <pre className="codeblock-pre" style={style}>
            {tokens.map((line, i) => (
              <span key={i} {...getLineProps({ line })} className="cb-line">
                <span className="cb-ln">{i + 1}</span>
                <span className="cb-code">
                  {line.map((token, k) => (
                    <span key={k} {...getTokenProps({ token })} />
                  ))}
                </span>
              </span>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  )
}
