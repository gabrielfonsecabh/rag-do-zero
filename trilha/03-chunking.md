# 03. Chunking

O modelo de embedding resume um texto inteiro num único vetor. Isso cria um dilema:

- **Trecho grande** mistura assuntos. O vetor vira uma média sem foco, e a pergunta específica não o encontra.
- **Trecho pequeno** perde contexto. "Até 10 dias" sozinho não diz que o assunto é venda de férias.

Chunking é escolher esse ponto de equilíbrio.

## As regras deste projeto

1. **Respeitar a estrutura.** O texto é dividido pelos títulos do markdown, e um trecho nunca junta duas seções.
2. **Levar o título junto.** Cada trecho guarda o caminho da seção ("Férias > Abono") e esse caminho entra no embedding. O título é a melhor pista de assunto que existe.
3. **Quebrar só o que precisa.** Seção que cabe no limite vira um trecho só. Seção maior quebra por parágrafo e, se ainda assim não couber, por frase.
4. **Sobrepor.** Ao quebrar, o fim de um trecho se repete no começo do próximo, para uma ideia que cruza o corte aparecer inteira em algum lugar.

## Experimente

```bash
python licoes/03_chunking.py                       # padrão: 800 caracteres, 150 de sobreposição
python licoes/03_chunking.py --max 200 --sobreposicao 40
```

Depois rode a avaliação (lição 06) com cada configuração. Chunking bom é o que faz o acerto subir, não o que parece mais elegante.

## Para documentos do mundo real

- **PDF:** extraia o texto antes (por exemplo com `pypdf` ou `pdfplumber`) e confira o resultado. Cabeçalho e rodapé repetidos em toda página poluem os trechos.
- **Tabelas:** costumam virar texto embaralhado. Transforme cada linha numa frase ("Hospedagem em capitais: até R$ 380 por noite") ou guarde a tabela inteira como um trecho.
- **Pontos de partida de tamanho:** de 300 a 1.000 caracteres para textos de política e manual, maior para textos narrativos. Meça com a lição 06.
