# 🖼️ AI Upscaler ALTA QUALIDADE

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-green.svg)](https://windows.microsoft.com)
[![License](https://img.shields.io/badge/License-Educational-orange.svg)](#licença)

Um upscaler de imagens focado em qualidade máxima usando técnicas comprovadas de IA e processamento avançado. Este script permite aumentar a resolução de imagens com foco em detalhes, texturas e nitidez realista.

## 📋 Índice

- [Características](#-características)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Solução de Problemas](#-solução-de-problemas)
- [Licença](#-licença)

## ✨ Características

### 🔬 Foco em Qualidade Máxima
- Upscaling de 4x usando modelos de IA comprovados
- Pós-processamento avançado para detalhes realistas
- Preservação de texturas e bordas
- Melhorias de contraste e nitidez

### 🧠 Técnicas Comprovadas
- **OpenCV EDSR**: Modelo de Super Resolução de última geração
- **PyTorch Multi-escala**: Upscaling em múltiplas etapas com realce
- **Processamento de Bordas Inteligente**: Realce seletivo de detalhes
- **Preservação de Textura**: Manutenção da qualidade original

### 🛠️ Pós-processamento Avançado
- Filtros de redução de ruído
- Realce de nitidez (Unsharp Masking)
- Ajustes de contraste e brilho
- Correção de gamma e CLAHE
- Redução de artefatos de upscale

## 🚀 Instalação

### Pré-requisitos

- **Sistema Operacional**: Windows 10/11
- **Python**: Versão 3.7 ou superior
  - Download: [python.org](https://python.org)
  - Alternativa: Microsoft Store

### Passos de Instalação

1. **Baixar o Script**
   ```bash
   # Clone o repositório ou baixe o arquivo ai_upscaler_hq.py
   # Salve em uma pasta conhecida (ex: C:\ai_upscaler)
   ```

2. **Verificar Python**
   ```cmd
   python --version
   # ou
   py --version
   ```

3. **Instalar Dependências**
   ```cmd
   # Abra CMD/PowerShell como Administrador
   pip install opencv-python opencv-contrib-python torch torchvision pillow numpy
   ```

4. **Executar o Script**
   ```cmd
   cd "C:\caminho\para\a\pasta\do\script"
   python ai_upscaler_hq.py
   ```

## 💡 Como Usar

### Execução Simples

```cmd
python ai_upscaler_hq.py
```

### Interface Interativa

```
🔧 HIGH QUALITY AI UPSCALER
   Foco em qualidade real, não velocidade
========================================
Arquivo: [arraste a imagem aqui ou digite o caminho]
```

### Resultado

- Pasta de saída: `High_Quality_4K/`
- Nome do arquivo: `HQ_nomeoriginal.extensao`

## 🛠️ Funcionalidades

### 1. Upscaling com OpenCV EDSR
- **Modelo EDSR x4**: Super resolução de alta qualidade
- **Pré-processamento**: Redução de ruído bilateral
- **Pós-processamento**: Sharpening e ajustes de qualidade
- **Redução de Artefatos**: Filtros medianos

### 2. Upscaling com PyTorch
- **Multi-escala**: Upscaling em 2 etapas (2x + 2x)
- **Realce de Bordas**: Detecção e melhoria de edges
- **Preservação de Textura**: Manutenção de detalhes originais
- **Melhorias Finais**: Correção de gamma e CLAHE

### 3. Processamento Avançado
- **Unsharp Masking**: Nitidez realista
- **Ajustes de Imagem**: Contraste, brilho, saturação
- **Filtros Adaptativos**: Redução de ruído inteligente
- **Correção de Cores**: Gamma e equalização

## 🔧 Tecnologias

| Biblioteca | Função | Versão |
|------------|---------|--------|
| **OpenCV** | Processamento de imagem e modelos DNN | Latest |
| **PyTorch** | Upscaling com redes neurais | Latest |
| **Pillow** | Manipulação avançada de imagens | Latest |
| **NumPy** | Processamento numérico | Latest |
| **urllib** | Download de modelos | Built-in |

## 🐛 Solução de Problemas

### ❌ Python não encontrado
```bash
# Soluções:
1. Instalar Python: https://python.org  
2. Microsoft Store: "Python 3.x"
3. Verificar PATH nas variáveis de ambiente
```

### ❌ Erro de dependências
```bash
# Execute:
pip install opencv-python opencv-contrib-python torch torchvision pillow numpy
```

### ❌ Modelos não baixam
```bash
# Soluções:
1. Verificar conexão com internet
2. Firewall/antivírus pode estar bloqueando
3. Tente baixar manualmente os modelos:
   - EDSR: https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb
   - FSRCNN: https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb
```

### ❌ CUDA Out of Memory
```bash
# Soluções:
1. Usar CPU: O script automaticamente usa CPU se CUDA não disponível
2. Fechar outros programas
3. Reiniciar o computador
```

## 📊 Recursos do Sistema

### Requisitos Mínimos
- **RAM**: 4GB (8GB recomendado)
- **Espaço em Disco**: 500MB para modelos e resultados
- **Processador**: Intel i5 ou equivalente
- **GPU**: Recomendado (CUDA) para PyTorch

### Compatibilidade
- ✅ Windows 10 (todas as versões)
- ✅ Windows 11 (todas as versões)  
- ✅ Linux (com ajustes)
- ❌ Windows 7/8 (não testado)

## 🔐 Segurança e Privacidade

### Características de Segurança
- **Offline**: Funciona sem internet após download dos modelos
- **Código Aberto**: Transparência total
- **Sem Telemetria**: Nenhum dado enviado externamente
- **Modelos Locais**: Todos os modelos são armazenados localmente

### Dados Coletados
- **Nenhum**: O software não coleta nem transmite dados pessoais
- **Modelos**: Baixados do GitHub (repositórios públicos)
- **Sem Analytics**: Nenhum rastreamento de uso

## 📈 Resultados Esperados

> **Resultados típicos**:
> - 📸 **Upscaling**: 4x de resolução original
> - 🎨 **Qualidade**: Preservação de detalhes e texturas
> - ⚡ **Tempo**: 30s-2min por imagem (dependendo do tamanho)
> - 💾 **Tamanho Final**: 16x maior que original (4x4x)

## 📜 Licença

Este projeto é fornecido **apenas para fins educacionais e de uso pessoal**. 

### Termos de Uso
- ✅ Uso pessoal e educacional
- ✅ Modificação para aprendizado
- ❌ Distribuição comercial
- ❌ Uso corporativo sem autorização

## ⚠️ Aviso Legal

### Importante
- 📋 **Backup**: Sempre faça backup das imagens originais
- 🔧 **Responsabilidade**: O uso é de sua inteira responsabilidade  
- ⚙️ **Qualidade**: Resultados podem variar conforme imagem original
- 🔒 **Segurança**: Execute apenas de fontes confiáveis

### Isenção de Responsabilidade
O desenvolvedor não se responsabiliza por perda de dados ou qualquer outro problema decorrente do uso deste software.

## 🤝 Contribuição

### Como Contribuir
1. 🍴 Fork o projeto
2. 🌟 Crie uma branch para sua feature
3. ✅ Teste suas modificações
4. 📝 Documente as mudanças
5. 🔄 Envie um Pull Request

### Diretrizes
- Mantenha o código limpo e documentado
- Teste com diferentes tipos de imagens
- Siga os padrões PEP 8 para Python
- Inclua exemplos de uso

## 📞 Suporte

### Canais de Suporte
- 📧 **Issues**: Use a seção Issues do GitHub
- 📖 **Documentação**: Consulte este README
- 🔍 **Debug**: Execute via CMD para ver logs detalhados

### FAQ
**P: O programa é seguro?**  
R: Sim, é código aberto e não envia dados externos.

**P: Funciona offline?**  
R: Sim, após baixar os modelos necessários.

**P: Qual a melhor qualidade?**  
R: OpenCV EDSR geralmente oferece melhor resultado.

---

<div align="center">

**Desenvolvido com ❤️ para entusiastas de qualidade de imagem**

[⬆️ Voltar ao topo](#-ai-upscaler-alta-qualidade)

</div>
