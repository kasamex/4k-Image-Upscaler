#!/usr/bin/env python3
"""
AI Upscaler ALTA QUALIDADE
Foca em qualidade real usando técnicas comprovadas
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import urllib.request

class HighQualityUpscaler:
    def __init__(self):
        self.models_dir = Path("hq_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # URLs que realmente funcionam
        self.working_urls = {
            'opencv_edsr_x4': 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb',
            'opencv_fsrcnn_x4': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb',
        }
        
        print("🔧 High Quality Upscaler inicializado")

    def check_opencv_dnn(self):
        """Verificar OpenCV DNN"""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            return True
        except:
            return False

    def download_opencv_model(self, model_name):
        """Baixar modelo OpenCV"""
        if model_name not in self.working_urls:
            return None
            
        model_path = self.models_dir / f"{model_name}.pb"
        if model_path.exists():
            return model_path
            
        try:
            print(f"Baixando {model_name}...")
            urllib.request.urlretrieve(self.working_urls[model_name], model_path)
            print(f"✅ {model_name} baixado")
            return model_path
        except:
            print(f"❌ Erro ao baixar {model_name}")
            return None

    def upscale_opencv_enhanced(self, input_path, output_path):
        """OpenCV com melhorias de qualidade"""
        if not self.check_opencv_dnn():
            return False, "OpenCV DNN não disponível"
            
        # Tentar EDSR primeiro (melhor qualidade)
        model_path = self.download_opencv_model('opencv_edsr_x4')
        if not model_path:
            return False, "Modelo EDSR não disponível"
            
        try:
            img = cv2.imread(str(input_path))
            if img is None:
                return False, "Imagem não carregada"
                
            print(f"Original: {img.shape[1]}x{img.shape[0]}")
            
            # Pré-processamento para melhor qualidade
            # 1. Redução de ruído
            img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
            
            # 2. OpenCV Super Resolution
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model_path))
            sr.setModel('edsr', 4)
            
            result = sr.upsample(img_denoised)
            
            # 3. Pós-processamento avançado
            result = self.post_process_advanced(result)
            
            print(f"Resultado: {result.shape[1]}x{result.shape[0]}")
            cv2.imwrite(str(output_path), result)
            
            return True, "OpenCV EDSR Enhanced: Sucesso"
            
        except Exception as e:
            return False, f"OpenCV Enhanced erro: {str(e)}"

    def post_process_advanced(self, image):
        """Pós-processamento avançado para melhorar qualidade"""
        
        # Converter para PIL para operações avançadas
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Unsharp masking (melhora detalhes)
        gaussian = pil_image.filter(ImageFilter.GaussianBlur(radius=2))
        unsharp_mask = Image.blend(pil_image, gaussian, -0.5)
        
        # 2. Ajustes de qualidade
        enhancer = ImageEnhance.Sharpness(unsharp_mask)
        enhanced = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.15)
        
        # 3. Redução de artefatos
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    def upscale_pytorch_improved(self, input_path, output_path):
        """PyTorch com técnicas de alta qualidade"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            img = cv2.imread(str(input_path))
            if img is None:
                return False, "Imagem não carregada"
                
            print(f"Original: {img.shape[1]}x{img.shape[0]}")
            
            # Converter para tensor
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                # Técnica multi-escala
                result = self.multi_scale_upscale(img_tensor, device)
                
                # Edge enhancement
                result = self.enhance_edges(result, device)
                
                # Texture preservation
                result = self.preserve_texture(result, img_tensor, device)
            
            # Converter resultado
            result_np = result.cpu().numpy().transpose(1, 2, 0)
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
            
            # Pós-processamento final
            final_result = self.final_enhancement(result_np)
            
            print(f"Resultado: {final_result.shape[1]}x{final_result.shape[0]}")
            cv2.imwrite(str(output_path), final_result)
            
            return True, "PyTorch Improved: Sucesso"
            
        except Exception as e:
            return False, f"PyTorch Improved erro: {str(e)}"

    def multi_scale_upscale(self, img_tensor, device):
        """Upscaling multi-escala para melhor qualidade"""
        
        # Escala 1: 2x com preservação de detalhes
        upscaled_2x = F.interpolate(img_tensor, scale_factor=2, mode='bicubic', align_corners=False)
        
        # Aplicar sharpening sutil
        sharp_kernel = torch.tensor([
            [[-0.25, -0.5, -0.25],
             [-0.5,   4,   -0.5],
             [-0.25, -0.5, -0.25]]
        ]).float().unsqueeze(0).to(device)
        
        enhanced_2x = []
        for i in range(3):
            channel = upscaled_2x[0, i:i+1, :, :].unsqueeze(0)
            enhanced = F.conv2d(channel, sharp_kernel, padding=1)
            enhanced_2x.append(enhanced.squeeze(0))
        
        enhanced_2x = torch.cat(enhanced_2x, dim=0).unsqueeze(0)
        enhanced_2x = torch.clamp(enhanced_2x, 0, 1)
        
        # Escala 2: mais 2x para 4x total
        upscaled_4x = F.interpolate(enhanced_2x, scale_factor=2, mode='bicubic', align_corners=False)
        
        return upscaled_4x

    def enhance_edges(self, img_tensor, device):
        """Realce de bordas inteligente"""
        
        # Detectar bordas
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().unsqueeze(0).to(device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().unsqueeze(0).to(device)
        
        gray = img_tensor.mean(dim=1, keepdim=True)
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Aplicar realce apenas nas bordas
        edge_mask = (edges > 0.1).float()
        
        # Kernel de sharpening
        sharp_kernel = torch.tensor([
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]]
        ]).float().unsqueeze(0).to(device)
        
        enhanced_channels = []
        for i in range(3):
            channel = img_tensor[0, i:i+1, :, :].unsqueeze(0)
            sharpened = F.conv2d(channel, sharp_kernel, padding=1)
            
            # Misturar original e sharpened baseado na máscara de bordas
            mask_3d = edge_mask.expand_as(channel)
            enhanced = channel * (1 - mask_3d * 0.3) + sharpened * (mask_3d * 0.3)
            enhanced_channels.append(enhanced.squeeze(0))
        
        return torch.cat(enhanced_channels, dim=0).unsqueeze(0)

    def preserve_texture(self, upscaled, original, device):
        """Preservar texturas da imagem original"""
        
        # Redimensionar original para comparação
        original_up = F.interpolate(original, size=upscaled.shape[2:], mode='bicubic', align_corners=False)
        
        # Calcular diferença de textura
        texture_diff = upscaled - original_up
        
        # Aplicar filtro passa-alta para preservar detalhes
        high_pass = torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ]).float().unsqueeze(0).to(device) / 8.0
        
        texture_enhanced = []
        for i in range(3):
            channel = texture_diff[0, i:i+1, :, :].unsqueeze(0)
            enhanced = F.conv2d(channel, high_pass, padding=1)
            texture_enhanced.append(enhanced.squeeze(0))
        
        texture_enhanced = torch.cat(texture_enhanced, dim=0).unsqueeze(0)
        
        # Combinar com imagem upscaled
        result = upscaled + texture_enhanced * 0.2
        
        return torch.clamp(result, 0, 1)

    def final_enhancement(self, image_np):
        """Melhorias finais para máxima qualidade"""
        
        # Converter para PIL
        pil_image = Image.fromarray(image_np)
        
        # 1. Correção de gamma para melhor contraste
        gamma = 1.1
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        gamma_corrected = cv2.LUT(np.array(pil_image), gamma_table)
        pil_image = Image.fromarray(gamma_corrected)
        
        # 2. Ajuste adaptativo de histograma (CLAHE em PIL)
        img_yuv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        pil_image = Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))
        
        # 3. Sharpening final sutil
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # 4. Ajuste final de contraste
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def process_high_quality(self, input_path, output_path):
        """Processamento com foco em máxima qualidade"""
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        print(f"\n🔧 Processamento alta qualidade: {input_path.name}")
        
        if not input_path.exists():
            return False, "Arquivo não encontrado"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Métodos focados em qualidade
        methods = [
            ("OpenCV EDSR Enhanced", lambda: self.upscale_opencv_enhanced(input_path, output_path)),
            ("PyTorch Improved", lambda: self.upscale_pytorch_improved(input_path, output_path))
        ]
        
        for method_name, method_func in methods:
            print(f"\n🚀 Tentando: {method_name}")
            success, message = method_func()
            
            if success:
                print(f"✅ {message}")
                
                # Mostrar estatísticas
                try:
                    img = cv2.imread(str(output_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        print(f"📈 Resultado: {w}x{h} ({size_mb:.1f}MB)")
                except:
                    pass
                
                return True, f"Sucesso com {method_name}"
            else:
                print(f"⚠️ {message}")
        
        return False, "Métodos de alta qualidade falharam"

def main():
    print("🔧 HIGH QUALITY AI UPSCALER")
    print("   Foco em qualidade real, não velocidade")
    print("="*40)
    
    upscaler = HighQualityUpscaler()
    
    input_file = input("Arquivo: ").strip().strip('"')
    if not input_file:
        return
        
    input_path = Path(input_file)
    if not input_path.exists():
        print("❌ Arquivo não encontrado")
        return
    
    output_dir = input_path.parent / "High_Quality_4K"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"HQ_{input_path.name}"
    
    success, message = upscaler.process_high_quality(input_path, output_path)
    
    if success:
        print(f"\n🎉 ALTA QUALIDADE ALCANÇADA!")
        print(f"📁 Salvo: {output_path}")
    else:
        print(f"\n❌ {message}")

if __name__ == "__main__":
    main()