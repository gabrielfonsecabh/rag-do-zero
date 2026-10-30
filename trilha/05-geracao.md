# 05. Geração com contexto

Com os trechos em mãos, o prompt tem três partes:

1. **Instrução de sistema:** responder só com base nos trechos, citar a fonte e admitir quando não sabe.
2. **Os trechos numerados**, cada um com arquivo e seção.
3. **A pergunta.**

O texto está em `python/rag/pipeline.py` (`SISTEMA` e `montar_prompt`) e é igual na versão TypeScript.

```bash
python licoes/05_rag_completo.py "Quantos dias de férias posso vender?"
```

Sem chave de API, a lição mostra o prompt exato em vez de chamar um modelo. Leia esse prompt. Se a resposta não está nele, o problema é a busca, e nenhum ajuste de modelo resolve.

## As três instruções que mais importam

- **"Use SOMENTE os trechos."** Sem isso o modelo completa com o que acha que sabe, e a resposta parece certa sem ser.
- **"Se não estiver nos trechos, diga que não encontrou."** Teste sempre com uma pergunta que a base não responde ("Qual é o plano de saúde?"). Um RAG que inventa nessa hora não é confiável nas outras.
- **"Cite o número do trecho."** Permite que a pessoa confira, e permite que você meça se a resposta veio mesmo do contexto.

## Escolhendo o gerador

```bash
GERACAO_PROVEDOR=anthropic ANTHROPIC_API_KEY=... python licoes/05_rag_completo.py "..."
GERACAO_PROVEDOR=openai OPENAI_API_KEY=... python licoes/05_rag_completo.py "..."

# grátis e local, com Ollama
GERACAO_PROVEDOR=openai OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  GERACAO_MODELO=llama3.2 python licoes/05_rag_completo.py "..."
```

Modelos menores e mais baratos costumam bastar para RAG, porque o trabalho difícil (achar a informação) já foi feito pela busca.
