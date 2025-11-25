import numpy as np
import pickle
import os
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class MLMovementAnalyzer:
    """
    Sistema de análise de movimento baseado em Machine Learning.
    Usa Random Forest para classificar qualidade das repetições.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Histórico de features para análise preditiva
        self.feature_history = deque(maxlen=50)
        self.rep_history = []
        
        # Métricas de sessão
        self.session_stats = {
            'total_reps': 0,
            'good_reps': 0,
            'average_depth': [],
            'average_lockout': [],
            'hip_sag_incidents': 0,
            'form_degradation': []
        }
        
        # Tentar carregar modelo pré-treinado
        self._load_model()
        
        # Se não existir, criar modelo com dados sintéticos
        if not self.is_trained:
            self._create_synthetic_training_data()
    
    def extract_features(self, landmarks, elbow_angle, hip_angle, 
                        min_angle=None, max_angle=None, stage="up"):
        """
        Extrai 15 features objetivas do movimento para análise ML
        """
        try:
            L = landmarks
            
            # 1-2: Ângulos principais
            features = [
                elbow_angle,
                hip_angle
            ]
            
            # 3-4: Amplitude de movimento
            depth = min_angle if min_angle is not None else elbow_angle
            lockout = max_angle if max_angle is not None else elbow_angle
            features.extend([depth, lockout])
            
            # 5-6: Simetria (comparar lados esquerdo/direito)
            left_vis = L[11].visibility  # LEFT_SHOULDER
            right_vis = L[12].visibility  # RIGHT_SHOULDER
            symmetry = abs(left_vis - right_vis)
            visibility_avg = (left_vis + right_vis) / 2
            features.extend([symmetry, visibility_avg])
            
            # 7-9: Alinhamento ombro-quadril-tornozelo
            side = "left" if left_vis >= right_vis else "right"
            sh_idx = 11 if side == "left" else 12
            hp_idx = 23 if side == "left" else 24
            an_idx = 27 if side == "left" else 28
            
            sh = L[sh_idx]
            hp = L[hp_idx]
            an = L[an_idx]
            
            # Desvio vertical normalizado
            vertical_alignment = abs((sh.y - hp.y) - (hp.y - an.y))
            # Desvio horizontal
            horizontal_spread = abs(sh.x - an.x)
            # Inclinação do tronco
            trunk_slope = abs(np.arctan2(hp.y - sh.y, hp.x - sh.x))
            
            features.extend([vertical_alignment, horizontal_spread, trunk_slope])
            
            # 10-11: Posição dos pulsos (importante para forma)
            wrist_idx = 15 if side == "left" else 16
            wr = L[wrist_idx]
            wrist_below_shoulder = max(0, wr.y - sh.y)
            wrist_shoulder_distance = np.sqrt((wr.x - sh.x)**2 + (wr.y - sh.y)**2)
            features.extend([wrist_below_shoulder, wrist_shoulder_distance])
            
            # 12-13: Estabilidade (variação angular)
            if len(self.feature_history) > 0:
                prev_elbow = self.feature_history[-1][0]
                prev_hip = self.feature_history[-1][1]
                elbow_stability = abs(elbow_angle - prev_elbow)
                hip_stability = abs(hip_angle - prev_hip)
            else:
                elbow_stability = 0
                hip_stability = 0
            features.extend([elbow_stability, hip_stability])
            
            # 14-15: Fase do movimento e velocidade
            stage_numeric = 1.0 if stage == "up" else 0.0
            movement_speed = elbow_stability  # proxy para velocidade
            features.extend([stage_numeric, movement_speed])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            # Retornar features neutras em caso de erro
            return np.zeros((1, 15))
    
    def _create_synthetic_training_data(self):
        """
        Cria dados de treino sintéticos baseados em regras biomecânicas
        Para simular padrões de movimento bom/médio/ruim
        """
        print("[ML] Criando modelo com dados sintéticos...")
        
        X_train = []
        y_train = []
        
        # Classe 0: EXCELENTE (80-100 pontos)
        for _ in range(200):
            features = [
                np.random.uniform(80, 100),   # elbow: profundo
                np.random.uniform(170, 180),  # hip: alinhado
                np.random.uniform(75, 95),    # depth: bom
                np.random.uniform(155, 175),  # lockout: completo
                np.random.uniform(0, 0.1),    # simetria: alta
                np.random.uniform(0.8, 1.0),  # visibilidade: boa
                np.random.uniform(0, 0.05),   # vertical_align: bom
                np.random.uniform(0.3, 0.6),  # horizontal_spread: adequado
                np.random.uniform(0, 0.2),    # trunk_slope: reto
                np.random.uniform(0.05, 0.15), # wrist_below
                np.random.uniform(0.2, 0.4),  # wrist_shoulder_dist
                np.random.uniform(0, 3),      # elbow_stability: suave
                np.random.uniform(0, 2),      # hip_stability: suave
                np.random.choice([0.0, 1.0]), # stage
                np.random.uniform(0, 3)       # movement_speed
            ]
            X_train.append(features)
            y_train.append(2)  # Classe EXCELENTE
        
        # Classe 1: BOM (60-80 pontos)
        for _ in range(200):
            features = [
                np.random.uniform(85, 110),   # elbow: ok
                np.random.uniform(160, 175),  # hip: leve queda
                np.random.uniform(90, 110),   # depth: médio
                np.random.uniform(145, 165),  # lockout: quase completo
                np.random.uniform(0.05, 0.2), # simetria: média
                np.random.uniform(0.6, 0.9),  # visibilidade: razoável
                np.random.uniform(0.03, 0.1), # vertical_align: ok
                np.random.uniform(0.25, 0.7), # horizontal_spread
                np.random.uniform(0.15, 0.35), # trunk_slope
                np.random.uniform(0.03, 0.18), # wrist_below
                np.random.uniform(0.15, 0.5), # wrist_shoulder_dist
                np.random.uniform(2, 7),      # elbow_stability
                np.random.uniform(1, 5),      # hip_stability
                np.random.choice([0.0, 1.0]),
                np.random.uniform(2, 7)
            ]
            X_train.append(features)
            y_train.append(1)  # Classe BOM
        
        # Classe 2: PRECISA MELHORAR (<60 pontos)
        for _ in range(200):
            features = [
                np.random.uniform(100, 140),  # elbow: raso
                np.random.uniform(140, 165),  # hip: queda significativa
                np.random.uniform(105, 130),  # depth: insuficiente
                np.random.uniform(130, 155),  # lockout: incompleto
                np.random.uniform(0.15, 0.4), # simetria: baixa
                np.random.uniform(0.4, 0.7),  # visibilidade: ruim
                np.random.uniform(0.08, 0.2), # vertical_align: ruim
                np.random.uniform(0.15, 0.9), # horizontal_spread: muito variado
                np.random.uniform(0.3, 0.7),  # trunk_slope: inclinado
                np.random.uniform(0, 0.08),   # wrist_below: inadequado
                np.random.uniform(0.1, 0.6),  # wrist_shoulder_dist
                np.random.uniform(5, 15),     # elbow_stability: instável
                np.random.uniform(3, 12),     # hip_stability: instável
                np.random.choice([0.0, 1.0]),
                np.random.uniform(5, 15)
            ]
            X_train.append(features)
            y_train.append(0)  # Classe PRECISA MELHORAR
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Normalizar e treinar
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        # Salvar modelo
        self._save_model()
        
        print(f"[ML] Modelo treinado: {len(X_train)} amostras")
        print(f"[ML] Acurácia no treino: {self.model.score(X_scaled, y_train):.2%}")
    
    def predict_quality(self, features):
        """
        Prediz qualidade da repetição (0=Ruim, 1=Bom, 2=Excelente)
        Retorna classe, probabilidades e score numérico
        """
        if not self.is_trained:
            return 1, [0.0, 1.0, 0.0], 70.0  # default
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Converter para score 0-100
        score = (
            probabilities[0] * 50 +    # Ruim: 0-50
            probabilities[1] * 75 +    # Bom: 50-85
            probabilities[2] * 95      # Excelente: 85-100
        )
        
        return int(prediction), probabilities, float(score)
    
    def analyze_rep(self, landmarks, elbow_angle, hip_angle, 
                   min_angle, max_angle, stage):
        """
        Análise completa de uma repetição usando ML
        """
        # Extrair features
        features = self.extract_features(
            landmarks, elbow_angle, hip_angle,
            min_angle, max_angle, stage
        )
        
        # Adicionar ao histórico
        self.feature_history.append(features[0])
        
        # Predição
        quality_class, probabilities, score = self.predict_quality(features)
        
        # Mapear classe para texto
        quality_map = {
            0: "Precisa Melhorar",
            1: "Bom",
            2: "Excelente"
        }
        
        grade_map = {
            0: "C",
            1: "B",
            2: "A+"
        }
        
        # Identificar pontos fracos (feature importance)
        weak_points = self._identify_weak_points(features[0])
        
        result = {
            'score': score,
            'grade': grade_map[quality_class],
            'quality': quality_map[quality_class],
            'probabilities': {
                'ruim': probabilities[0],
                'bom': probabilities[1],
                'excelente': probabilities[2]
            },
            'weak_points': weak_points,
            'features': features[0].tolist()
        }
        
        # Atualizar estatísticas
        self._update_session_stats(result, min_angle, max_angle, hip_angle)
        
        return result
    
    def _identify_weak_points(self, features):
        """
        Identifica aspectos específicos que precisam melhorar
        usando importância das features
        """
        issues = []
        
        elbow, hip, depth, lockout = features[0:4]
        symmetry, visibility = features[4:6]
        alignment_v, alignment_h, trunk = features[6:9]
        stability_elbow, stability_hip = features[11:13]
        
        # Análise objetiva baseada em thresholds estatísticos
        if depth > 100:
            issues.append("PROFUNDIDADE INSUFICIENTE")
        if lockout < 150:
            issues.append("EXTENSÃO INCOMPLETA")
        if hip < 165:
            issues.append("QUEDA DE QUADRIL")
        if symmetry > 0.2:
            issues.append("ASSIMETRIA")
        if stability_elbow > 10 or stability_hip > 8:
            issues.append("MOVIMENTO INSTÁVEL")
        if trunk > 0.4:
            issues.append("TRONCO INCLINADO")
        
        return issues if issues else ["OK"]
    
    def _update_session_stats(self, result, min_angle, max_angle, hip_angle):
        """Atualiza estatísticas da sessão"""
        self.session_stats['total_reps'] += 1
        if result['score'] >= 75:
            self.session_stats['good_reps'] += 1
        
        if min_angle:
            self.session_stats['average_depth'].append(min_angle)
        if max_angle:
            self.session_stats['average_lockout'].append(max_angle)
        if hip_angle < 165:
            self.session_stats['hip_sag_incidents'] += 1
        
        # Detectar degradação de forma (últimas 5 reps)
        self.rep_history.append(result['score'])
        if len(self.rep_history) >= 5:
            recent_trend = np.mean(self.rep_history[-5:])
            self.session_stats['form_degradation'].append(recent_trend)
    
    def get_predictive_suggestions(self):
        """
        Análise preditiva: sugere melhorias baseadas em tendências
        """
        suggestions = []
        
        if len(self.rep_history) < 3:
            return ["Continue praticando para análise preditiva"]
        
        recent_scores = self.rep_history[-10:]
        avg_score = np.mean(recent_scores)
        
        # Tendência de melhora/piora
        if len(recent_scores) >= 5:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half < first_half - 5:
                suggestions.append("⚠️ FADIGA DETECTADA - Considere descanso")
            elif second_half > first_half + 5:
                suggestions.append("📈 MELHORA CONSISTENTE - Ótimo progresso!")
        
        # Análise de profundidade
        if self.session_stats['average_depth']:
            avg_depth = np.mean(self.session_stats['average_depth'])
            if avg_depth > 100:
                suggestions.append("💡 Aumente a profundidade - Desça mais 10-15°")
        
        # Análise de extensão
        if self.session_stats['average_lockout']:
            avg_lockout = np.mean(self.session_stats['average_lockout'])
            if avg_lockout < 155:
                suggestions.append("💡 Melhore a extensão - Estenda completamente os braços")
        
        # Análise de quadril
        hip_rate = (self.session_stats['hip_sag_incidents'] / 
                   max(1, self.session_stats['total_reps']))
        if hip_rate > 0.3:
            suggestions.append("🎯 Foco no core - Contraia abdômen durante todo movimento")
        
        # Score médio
        if avg_score < 65:
            suggestions.append("📚 Reveja a técnica - Qualidade > Quantidade")
        elif avg_score >= 85:
            suggestions.append("🏆 Excelente técnica! Aumente a dificuldade")
        
        return suggestions if suggestions else ["✅ Mantenha a boa forma!"]
    
    def get_session_report(self):
        """Gera relatório completo da sessão"""
        if self.session_stats['total_reps'] == 0:
            return "Nenhuma repetição registrada"
        
        good_rate = (self.session_stats['good_reps'] / 
                    self.session_stats['total_reps'] * 100)
        
        avg_score = np.mean(self.rep_history) if self.rep_history else 0
        
        report = {
            'total_reps': self.session_stats['total_reps'],
            'good_reps': self.session_stats['good_reps'],
            'success_rate': good_rate,
            'average_score': avg_score,
            'suggestions': self.get_predictive_suggestions()
        }
        
        return report
    
    def _save_model(self):
        """Salva modelo treinado"""
        try:
            model_dir = "c:\\VC_proj\\models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{model_dir}\\pushup_analyzer.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"[ML] Modelo salvo em {model_dir}")
        except Exception as e:
            print(f"[ML] Erro ao salvar modelo: {e}")
    
    def _load_model(self):
        """Carrega modelo pré-treinado"""
        try:
            model_path = "c:\\VC_proj\\models\\pushup_analyzer.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.is_trained = True
                print(f"[ML] Modelo carregado de {model_path}")
        except Exception as e:
            print(f"[ML] Não foi possível carregar modelo: {e}")