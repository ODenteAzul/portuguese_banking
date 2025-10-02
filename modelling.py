import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, matthews_corrcoef,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


import shap

import warnings
warnings.filterwarnings('ignore')


class BankMarketingPipeline:
    """
    Pipeline completo de modelagem para Bank Marketing Dataset.

    VERSÃO CORRIGIDA: Business metrics agora projetam para população total.

    Estrutura:
    1. Preparação e split estratificado
    2. Baseline models (múltiplos algoritmos)
    3. Feature selection
    4. Hyperparameter tuning
    5. Modelo final e interpretabilidade
    6. Business metrics e ROI (CORRIGIDO)
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            test_size: float = 0.2,
            random_state: int = 42):
        """
        Inicializa o pipeline com dados tratados.

        Args:
            X: Features (já tratadas e encoded)
            y: Target (0/1)
            test_size: proporção do test set
            random_state: seed para reprodutibilidade
        """
        self.X = X
        self.y = y
        self.random_state = random_state

        # Guardar tamanho da população total para business metrics no final
        self.total_population = len(X)

        # Split estratificado (mantém proporção de classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(
            f""" Dataset dividido:
              Train={len(self.X_train)} | Test={len(self.X_test)}""")
        print(
            f""" Distribuição train:
              {dict(pd.Series(self.y_train).value_counts())}""")
        print(
            f""" Distribuição test:
              {dict(pd.Series(self.y_test).value_counts())}""")
        print(
            f""" População total:
              {self.total_population}""")

        # Dicionário para armazenar resultados
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None

    # BASELINE para availiação inicial e contrle

    def train_baseline_models(self) -> pd.DataFrame:
        """
        Treina múltiplos modelos baseline para comparação.

        Returns:
            DataFrame com métricas de todos os modelos

        Justificativa da escolha de modelos:
        - Logistic Regression: baseline interpretável, rápido
        - Decision Tree: não-linear, fácil explicar para negócio
        - Random Forest: robusto, lida bem com features correlacionadas
        - Gradient Boosting: geralmente melhor performance
        - XGBoost: state-of-art, lida bem com desbalanceamento
        - LightGBM: rápido, eficiente para grandes datasets
        """
        print("\n" + "="*70)
        print("TREINAMENTO DE BASELINE MODELS")
        print("="*70 + "\n")

        # Definir modelos
        # usar class_weight='balanced'
        # para lidar com desbalanceamento
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=50,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=50,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=len(
                    self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1]),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }

        results_list = []

        for name, model in models.items():
            print(f"Treinando {name}...")

            # Treinar modelo
            model.fit(self.X_train, self.y_train)

            # Predições
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Calcular métricas
            # Para dataset desbalanceado, F1, AUC-ROC e MCC são mais importantes que Accuracy
            metrics = {
                'Model': name,
                'Accuracy': np.mean(y_pred == self.y_test),
                'Precision': classification_report(self.y_test,
                                                   y_pred,
                                                   output_dict=True)['1']['precision'],
                'Recall': classification_report(self.y_test,
                                                y_pred,
                                                output_dict=True)['1']['recall'],
                'F1-Score': f1_score(self.y_test, y_pred),
                'AUC-ROC': roc_auc_score(self.y_test, y_pred_proba),
                'AUC-PR': average_precision_score(self.y_test, y_pred_proba),
                'MCC': matthews_corrcoef(self.y_test, y_pred)
            }

            results_list.append(metrics)

            # Armazenar modelo
            self.models[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': metrics
            }

            print(
                f"   F1-Score: {metrics['F1-Score']:.4f} | AUC-ROC: {metrics['AUC-ROC']:.4f}")

        # Criar DataFrame de resultados
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('F1-Score', ascending=False)

        print("\n" + "="*70)
        print("RESULTADOS DOS BASELINE MODELS (ordenado por F1-Score)")
        print("="*70)
        print(results_df.to_string(index=False, float_format='%.4f'))
        print()

        # Identificar melhor modelo
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        print(f"Melhor modelo baseline: {best_model_name}")
        print(f"   F1-Score: {self.best_model['metrics']['F1-Score']:.4f}")
        print(f"   AUC-ROC: {self.best_model['metrics']['AUC-ROC']:.4f}")

        return results_df

    def plot_baseline_comparison(self, save: bool = False):
        """
        Plota comparação visual dos baseline models.
        """
        if not self.models:
            print(" Execute train_baseline_models() primeiro!")
            return

        metrics_to_plot = ['Accuracy', 'Precision',
                           'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            model_names = []
            metric_values = []

            for name, data in self.models.items():
                model_names.append(name)
                metric_values.append(data['metrics'][metric])

            # Ordenar por valor
            sorted_pairs = sorted(
                zip(model_names, metric_values), key=lambda x: x[1], reverse=True)
            model_names, metric_values = zip(*sorted_pairs)

            # Cores: melhor modelo em verde
            colors = ['#06A77D' if i ==
                      0 else '#2E86AB' for i in range(len(model_names))]

            bars = ax.barh(model_names, metric_values,
                           color=colors, alpha=0.8, edgecolor='black')
            ax.set_xlabel(metric, fontsize=11)
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_xlim(0, 1)

            # Adicionar valores
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        if save:
            plt.savefig('baseline_comparison.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print(" Comparação de baseline models plotada")

    def plot_confusion_matrices(self, save: bool = False):
        """
        Plota matrizes de confusão de todos os modelos.
        """
        if not self.models:
            print(" Execute train_baseline_models() primeiro!")
            return

        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, (name, data) in enumerate(self.models.items()):
            ax = axes[idx]

            cm = confusion_matrix(self.y_test, data['y_pred'])

            # Normalizar para percentuais
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # Criar anotações
            annot = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                        cbar=False, square=True, linewidths=1,
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])

            ax.set_xlabel('Predição', fontsize=10)
            ax.set_ylabel('Real', fontsize=10)
            ax.set_title(f'{name}\nF1={data["metrics"]["F1-Score"]:.3f}',
                         fontsize=11, fontweight='bold')

        # Remover subplots vazios
        for idx in range(len(self.models), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(" Matrizes de confusão plotadas")

    def plot_roc_curves(self, save: bool = False):
        """
        Plota curvas ROC de todos os modelos em um único gráfico.
        """
        if not self.models:
            print(" Execute train_baseline_models() primeiro!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === ROC Curve ===
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))

        for (name, data), color in zip(self.models.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, data['y_pred_proba'])
            auc = data['metrics']['AUC-ROC']

            ax1.plot(fpr, tpr, linewidth=2,
                     label=f'{name} (AUC={auc:.3f})', color=color)

        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2,
                 label='Random (AUC=0.500)')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves - Comparação de Modelos',
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # === Precision-Recall Curve ===
        for (name, data), color in zip(self.models.items(), colors):
            precision, recall, _ = precision_recall_curve(
                self.y_test, data['y_pred_proba'])
            auc_pr = data['metrics']['AUC-PR']

            ax2.plot(recall, precision, linewidth=2,
                     label=f'{name} (AUC={auc_pr:.3f})', color=color)

        # Linha baseline (proporção da classe positiva)
        baseline = self.y_test.sum() / len(self.y_test)
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
                    label=f'Baseline (y={baseline:.3f})')

        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curves',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(" Curvas ROC e Precision-Recall plotadas")

    # ========================================================================
    # 2. FEATURE SELECTION
    # ========================================================================

    def feature_selection_analysis(self, k_features: int = 20, method: str = 'all') -> pd.DataFrame:
        """
        Analisa importância de features usando múltiplos métodos.

        Args:
            k_features: número de features a selecionar
            method: 'mutual_info', 'tree_based', 'rfe', 'all'

        Returns:
            DataFrame com ranking de features

        Justificativa:
        - Mutual Information: captura relações não-lineares
        - Tree-based: importância via Random Forest (padrão ouro)
        - RFE: seleção recursiva com Logistic Regression
        """
        print("\n" + "="*70)
        print("FEATURE SELECTION ANALYSIS")
        print("="*70 + "\n")

        feature_scores = pd.DataFrame({'feature': self.X_train.columns})

        # 1. Mutual Information
        if method in ['mutual_info', 'all']:
            print("Calculando Mutual Information...")
            mi_scores = mutual_info_classif(
                self.X_train, self.y_train, random_state=self.random_state)
            feature_scores['mi_score'] = mi_scores
            feature_scores['mi_rank'] = feature_scores['mi_score'].rank(
                ascending=False)

        # 2. Tree-based (Random Forest)
        if method in ['tree_based', 'all']:
            print("Calculando importância via Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            feature_scores['rf_importance'] = rf.feature_importances_
            feature_scores['rf_rank'] = feature_scores['rf_importance'].rank(
                ascending=False)

        # 3. RFE (Recursive Feature Elimination)
        if method in ['rfe', 'all']:
            print("Executando RFE...")
            lr = LogisticRegression(
                max_iter=1000, random_state=self.random_state)
            rfe = RFE(estimator=lr, n_features_to_select=k_features)
            rfe.fit(self.X_train, self.y_train)
            feature_scores['rfe_rank'] = rfe.ranking_
            feature_scores['rfe_selected'] = rfe.support_

        # Calcular rank médio
        rank_cols = [col for col in feature_scores.columns if 'rank' in col]
        if rank_cols:
            feature_scores['avg_rank'] = feature_scores[rank_cols].mean(axis=1)
            feature_scores = feature_scores.sort_values('avg_rank')

        print(f"\nTOP {k_features} FEATURES MAIS IMPORTANTES")
        print("="*70)
        print(feature_scores.head(k_features).to_string(
            index=False, float_format='%.4f'))
        print()

        self.feature_importance = feature_scores

        return feature_scores

    def plot_feature_importance(self, top_n: int = 20, save: bool = False):
        """
        Plota feature importance dos diferentes métodos.
        """
        if self.feature_importance is None:
            print(" Execute feature_selection_analysis() primeiro!")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        # 1. Mutual Information
        if 'mi_score' in self.feature_importance.columns:
            ax = axes[0]
            top_features = self.feature_importance.nlargest(top_n, 'mi_score')

            bars = ax.barh(top_features['feature'], top_features['mi_score'],
                           color='#2E86AB', alpha=0.8, edgecolor='black')
            ax.set_xlabel('MI Score', fontsize=11)
            ax.set_title('Mutual Information', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.invert_yaxis()

        # 2. Random Forest Importance
        if 'rf_importance' in self.feature_importance.columns:
            ax = axes[1]
            top_features = self.feature_importance.nlargest(
                top_n, 'rf_importance')

            bars = ax.barh(top_features['feature'], top_features['rf_importance'],
                           color='#06A77D', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Random Forest Importance',
                         fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.invert_yaxis()

        # 3. Average Rank
        if 'avg_rank' in self.feature_importance.columns:
            ax = axes[2]
            top_features = self.feature_importance.nsmallest(top_n, 'avg_rank')

            bars = ax.barh(top_features['feature'], 1/top_features['avg_rank'],
                           color='#A23B72', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Importance (1/Avg Rank)', fontsize=11)
            ax.set_title('Consensus Ranking', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.invert_yaxis()

        plt.tight_layout()
        if save:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(" Feature importance plotada")

    # ========================================================================
    # 3. HYPERPARAMETER TUNING
    # ========================================================================

    def hyperparameter_tuning(
        self,
        model_name: str = 'XGBoost',
        method: str = 'random',
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict:
        """
        Otimiza hiperparâmetros do melhor modelo usando RandomizedSearchCV ou GridSearchCV.

        Args:
            model_name: nome do modelo a otimizar
            method: 'random' ou 'grid'
            n_iter: número de iterações (apenas para random search)
            cv: número de folds para cross-validation

        Returns:
            Dicionário com melhor modelo e parâmetros
        """
        print("\n" + "="*70)
        print(f"HYPERPARAMETER TUNING - {model_name}")
        print("="*70 + "\n")

        # Definir param grids por modelo
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63, 127],
                'min_child_samples': [10, 20, 50],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [10, 20, 50, 100],
                'min_samples_leaf': [1, 2, 4, 10],
                'max_features': ['sqrt', 'log2', 0.5]
            }
        }

        # Obter modelo base
        if model_name == 'XGBoost':
            base_model = XGBClassifier(
                scale_pos_weight=len(
                    self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1]),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_name == 'LightGBM':
            base_model = LGBMClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        elif model_name == 'Random Forest':
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            print(f" Modelo '{model_name}' não suportado para tuning")
            return None

        # Configurar search
        scoring = 'f1'  # importante para dataset desbalanceado
        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state)

        if method == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grids[model_name],
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_splitter,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:  # grid
            search = GridSearchCV(
                base_model,
                param_grid=param_grids[model_name],
                scoring=scoring,
                cv=cv_splitter,
                n_jobs=-1,
                verbose=1
            )

        print(
            f"Iniciando {method} search com {n_iter if method == 'random' else 'all'} iterações...")
        print(f"Otimizando para: {scoring}")
        print(f"Cross-validation: {cv} folds estratificados\n")

        search.fit(self.X_train, self.y_train)

        print("\n" + "="*70)
        print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
        print("="*70)
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"\nMelhor F1-Score (CV): {search.best_score_:.4f}")

        # Avaliar no test set
        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"\nPerformance no Test Set:")
        print(f"   F1-Score: {test_f1:.4f}")
        print(f"   AUC-ROC: {test_auc:.4f}")
        print()

        return {
            'model': best_model,
            'best_params': search.best_params_,
            'cv_score': search.best_score_,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    # ========================================================================
    # 4. INTERPRETABILIDADE (SHAP)
    # ========================================================================

    def explain_model_shap(self, model, max_display: int = 20, save: bool = False):
        """
        Gera explicações SHAP para interpretabilidade do modelo.

        Args:
            model: modelo treinado
            max_display: número máximo de features a mostrar
            save: salvar figuras

        SHAP (SHapley Additive exPlanations):
        - Mostra contribuição de cada feature para cada predição
        - Baseado em teoria dos jogos (Shapley values)
        - Método state-of-art para interpretabilidade
        """
        print("\n" + "=" * 70)
        print("INTERPRETABILIDADE DO MODELO (SHAP)")
        print("=" * 70 + "\n")

        print("Calculando SHAP values (pode demorar alguns minutos)...")

        # Criar explainer
        # Para tree-based models, usar TreeExplainer (mais rápido)
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            # Para outros modelos, usar KernelExplainer (mais lento)
            explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(self.X_train, 100))

        # Calcular SHAP values (usar amostra para acelerar)
        sample_size = min(1000, len(self.X_test))
        X_sample = self.X_test.sample(
            n=sample_size, random_state=self.random_state)
        shap_values = explainer.shap_values(X_sample)

        # Se retornar lista (binary classification), pegar classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        print(" SHAP values calculados\n")

        # 1. Summary Plot (Feature Importance global)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample,
                          max_display=max_display, show=False)
        plt.title('SHAP Summary Plot - Importância Global das Features',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save:
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Bar Plot (Feature Importance - valores absolutos médios)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar",
                          max_display=max_display, show=False)
        plt.title('SHAP Feature Importance - Impacto Médio Absoluto',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save:
            plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(" Gráficos SHAP gerados")
        print("\nInterpretação:")
        print("   - Cor VERMELHA: valor alto da feature")
        print("   - Cor AZUL: valor baixo da feature")
        print("   - Eixo X: impacto na predição (positivo = aumenta chance de conversão)")
        print("   - Quanto mais à direita, maior o impacto positivo")

    # ========================================================================
    # 5. BUSINESS METRICS & ROI (VERSÃO CORRIGIDA)
    # ========================================================================

    def calculate_business_metrics(
        self,
        y_pred_proba: np.ndarray,
        cost_per_contact: float = 5.0,
        revenue_per_conversion: float = 200.0,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calcula métricas de negócio e ROI da campanha.

        VERSÃO CORRIGIDA: Projeta resultados do test set para população total.

        Args:
            y_pred_proba: probabilidades preditas
            cost_per_contact: custo por contato (ligação)
            revenue_per_conversion: receita por conversão (depósito)
            threshold: threshold de decisão

        Returns:
            Dicionário com métricas de negócio

        Justificativa:
        Em marketing direto, queremos maximizar ROI, não apenas accuracy.
        O modelo permite priorizar clientes com maior propensão.

        CORREÇÃO: Anteriormente usava apenas test set (9k clientes).
        Agora projeta para população total (45k clientes) para ROI realista.
        """

        print("\n" + "="*70)
        print("BUSINESS METRICS & ROI ANALYSIS")
        print("="*70 + "\n")

        # ===================================================================
        # CENÁRIO 1: BASELINE (Contatar TODOS da população)
        # ===================================================================

        # Usar população TOTAL, não apenas test set
        total_customers = self.total_population

        # Projetar taxa de conversão do test set para população
        test_conversion_rate = self.y_test.sum() / len(self.y_test)
        total_conversions = int(total_customers * test_conversion_rate)

        baseline_cost = total_customers * cost_per_contact
        baseline_revenue = total_conversions * revenue_per_conversion
        baseline_profit = baseline_revenue - baseline_cost
        baseline_roi = (baseline_profit / baseline_cost) * 100

        print("CENÁRIO BASELINE (Contatar todos da população):")
        print(f"   Total de clientes: {total_customers:,}")
        print(
            f"   Taxa de conversão estimada: {test_conversion_rate*100:.2f}%")
        print(f"   Conversões estimadas: {total_conversions:,}")
        print(f"   Custo total: R$ {baseline_cost:,.2f}")
        print(f"   Receita total: R$ {baseline_revenue:,.2f}")
        print(f"   Lucro: R$ {baseline_profit:,.2f}")
        print(f"   ROI: {baseline_roi:.2f}%")

        # ===================================================================
        # CENÁRIO 2: USAR MODELO (projetar test set para população)
        # ===================================================================

        y_pred = (y_pred_proba >= threshold).astype(int)

        # Métricas no test set
        true_positives_test = ((y_pred == 1) & (self.y_test == 1)).sum()
        false_positives_test = ((y_pred == 1) & (self.y_test == 0)).sum()
        false_negatives_test = ((y_pred == 0) & (self.y_test == 1)).sum()

        contacted_test = true_positives_test + false_positives_test
        conversions_test = true_positives_test

        # Taxa de precisão no test set
        precision_test = conversions_test / contacted_test if contacted_test > 0 else 0
        contact_rate_test = contacted_test / len(self.y_test)

        # PROJETAR para população total
        contacted_total = int(total_customers * contact_rate_test)
        conversions_total = int(contacted_total * precision_test)
        missed_conversions_total = total_conversions - conversions_total

        model_cost = contacted_total * cost_per_contact
        model_revenue = conversions_total * revenue_per_conversion
        model_profit = model_revenue - model_cost
        model_roi = (model_profit / model_cost) * 100 if model_cost > 0 else 0

        # Receita perdida (conversões que perdemos por não contatar)
        opportunity_cost = missed_conversions_total * revenue_per_conversion

        print(f"\nCENÁRIO COM MODELO (Threshold = {threshold}):")
        print(
            f"   Clientes contatados: {contacted_total:,} ({contact_rate_test*100:.1f}% da população)")
        print(f"   Conversões obtidas: {conversions_total:,}")
        print(f"   Conversões perdidas: {missed_conversions_total:,}")
        print(
            f"   Taxa de conversão: {precision_test*100:.2f}% (vs {test_conversion_rate*100:.2f}% baseline)")
        print(f"   Custo total: R$ {model_cost:,.2f}")
        print(f"   Receita total: R$ {model_revenue:,.2f}")
        print(f"   Lucro: R$ {model_profit:,.2f}")
        print(f"   ROI: {model_roi:.2f}%")
        print(f"   Custo de oportunidade: R$ {opportunity_cost:,.2f}")

        # Comparação
        cost_reduction = ((baseline_cost - model_cost) / baseline_cost) * 100
        profit_increase = model_profit - baseline_profit
        roi_increase = model_roi - baseline_roi

        # Economias absolutas
        cost_saved = baseline_cost - model_cost
        revenue_lost = baseline_revenue - model_revenue

        print(f"\nIMPACTO DO MODELO:")
        print(
            f"   Redução de custos: {cost_reduction:.1f}% (R$ {cost_saved:,.2f} economizados)")
        print(f"   Perda de receita: R$ {revenue_lost:,.2f}")
        print(f"   Variação de lucro: R$ {profit_increase:,.2f}")
        print(f"   Variação de ROI: {roi_increase:+.2f} pontos percentuais")

        if profit_increase > 0:
            print(
                f"\n MODELO VIÁVEL - Aumenta lucro em R$ {profit_increase:,.2f}")
        else:
            print(
                f"\n  MODELO REDUZ LUCRO EM R$ {abs(profit_increase):,.2f}")
            print("    Cenários onde modelo agrega valor:")

            # Calcular breakeven de custo
            breakeven_cost = revenue_per_conversion * \
                precision_test / (1 + baseline_roi/100)
            print(f"    - Custo por contato > R$ {breakeven_cost:.2f}")
            print(
                f"    - Capacidade limitada (< {contacted_total:,} contatos)")
            print(f"    - Usar modelo para TIMING/PERSONALIZAÇÃO, não seleção")

        return {
            'baseline': {
                'population': total_customers,
                'contacted': total_customers,
                'conversions': total_conversions,
                'conversion_rate': test_conversion_rate,
                'cost': baseline_cost,
                'revenue': baseline_revenue,
                'profit': baseline_profit,
                'roi': baseline_roi
            },
            'model': {
                'contacted': contacted_total,
                'contact_rate': contact_rate_test,
                'conversions': conversions_total,
                'precision': precision_test,
                'missed': missed_conversions_total,
                'cost': model_cost,
                'revenue': model_revenue,
                'profit': model_profit,
                'roi': model_roi,
                'opportunity_cost': opportunity_cost
            },
            'impact': {
                'cost_reduction_pct': cost_reduction,
                'cost_saved': cost_saved,
                'revenue_lost': revenue_lost,
                'profit_increase': profit_increase,
                'roi_increase': roi_increase,
                'viable': profit_increase > 0
            }
        }

    def optimize_threshold_for_profit(
        self,
        y_pred_proba: np.ndarray,
        cost_per_contact: float = 5.0,
        revenue_per_conversion: float = 200.0,
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        step: float = 0.01
    ) -> Dict:
        """
        Otimiza threshold para MAXIMIZAR LUCRO.

        VERSÃO CORRIGIDA: Projeta para população total.

        Args:
            y_pred_proba: Probabilidades preditas
            cost_per_contact: Custo por ligação
            revenue_per_conversion: Receita por conversão
            threshold_range: Range de thresholds a testar
            step: Passo entre thresholds

        Returns:
            Dict com threshold ótimo e métricas

        Justificativa:
        F1-Score não considera custos/receitas reais.
        Para negócio, maximizar LUCRO > maximizar F1.
        """
        print("\n" + "=" * 70)
        print("OTIMIZAÇÃO DE THRESHOLD PARA MAXIMIZAÇÃO DE LUCRO")
        print("=" * 70 + "\n")

        thresholds = np.arange(threshold_range[0], threshold_range[1], step)

        # População total para projeções
        total_customers = self.total_population
        test_conversion_rate = self.y_test.sum() / len(self.y_test)
        total_conversions_baseline = int(
            total_customers * test_conversion_rate)

        baseline_cost = total_customers * cost_per_contact
        baseline_revenue = total_conversions_baseline * revenue_per_conversion
        baseline_profit = baseline_revenue - baseline_cost

        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calcular métricas de classificação no test set
            tp = ((y_pred == 1) & (self.y_test == 1)).sum()
            fp = ((y_pred == 1) & (self.y_test == 0)).sum()
            fn = ((y_pred == 0) & (self.y_test == 1)).sum()
            tn = ((y_pred == 0) & (self.y_test == 0)).sum()

            # Taxas no test set
            contacted_test = tp + fp
            conversions_test = tp

            precision_test = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_test = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision_test * recall_test) / (precision_test +
                                                       recall_test) if (precision_test + recall_test) > 0 else 0

            contact_rate = contacted_test / len(self.y_test)

            # PROJETAR para população total
            contacted_total = int(total_customers * contact_rate)
            conversions_total = int(contacted_total * precision_test)
            missed_total = total_conversions_baseline - conversions_total

            # Métricas de negócio
            cost = contacted_total * cost_per_contact
            revenue = conversions_total * revenue_per_conversion
            profit = revenue - cost
            roi = (profit / cost * 100) if cost > 0 else 0

            results.append({
                'threshold': threshold,
                'contacted': contacted_total,
                'contact_rate': contact_rate,
                'conversions': conversions_total,
                'missed': missed_total,
                'cost': cost,
                'revenue': revenue,
                'profit': profit,
                'roi': roi,
                'precision': precision_test,
                'recall': recall_test,
                'f1': f1
            })

        # Converter para DataFrame
        df_results = pd.DataFrame(results)

        # Encontrar thresholds ótimos
        optimal_profit_idx = df_results['profit'].idxmax()
        optimal_profit = df_results.loc[optimal_profit_idx]

        optimal_roi_idx = df_results['roi'].idxmax()
        optimal_roi = df_results.loc[optimal_roi_idx]

        optimal_f1_idx = df_results['f1'].idxmax()
        optimal_f1 = df_results.loc[optimal_f1_idx]

        # Exibir resultados
        print("THRESHOLD ÓTIMO PARA LUCRO:")
        print(f"  Threshold: {optimal_profit['threshold']:.2f}")
        print(f"  Lucro: R$ {optimal_profit['profit']:,.2f}")
        print(f"  ROI: {optimal_profit['roi']:.1f}%")
        print(f"  F1-Score: {optimal_profit['f1']:.3f}")
        print(
            f"  Contatos: {optimal_profit['contacted']:.0f} ({optimal_profit['contact_rate']*100:.1f}% da população)")
        print(
            f"  Conversões: {optimal_profit['conversions']:.0f} ({(optimal_profit['conversions']/total_conversions_baseline)*100:.1f}% do total)")

        print("\nTHRESHOLD ÓTIMO PARA ROI:")
        print(f"  Threshold: {optimal_roi['threshold']:.2f}")
        print(f"  ROI: {optimal_roi['roi']:.1f}%")
        print(f"  Lucro: R$ {optimal_roi['profit']:,.2f}")
        print(f"  F1-Score: {optimal_roi['f1']:.3f}")

        print("\nTHRESHOLD ÓTIMO PARA F1 (referência):")
        print(f"  Threshold: {optimal_f1['threshold']:.2f}")
        print(f"  F1-Score: {optimal_f1['f1']:.3f}")
        print(f"  Lucro: R$ {optimal_f1['profit']:,.2f}")

        # Comparação com baseline
        profit_diff_optimal = optimal_profit['profit'] - baseline_profit
        profit_diff_f1 = optimal_f1['profit'] - baseline_profit

        print(f"\nCOMPARAÇÃO COM BASELINE (Lucro: R$ {baseline_profit:,.2f}):")
        print(
            f"  Threshold ótimo lucro: {profit_diff_optimal:+,.2f} ({(profit_diff_optimal/baseline_profit)*100:+.1f}%)")
        print(
            f"  Threshold ótimo F1: {profit_diff_f1:+,.2f} ({(profit_diff_f1/baseline_profit)*100:+.1f}%)")

        if profit_diff_optimal > 0:
            print(
                f"\n Modelo VIÁVEL - Threshold {optimal_profit['threshold']:.2f} gera R$ {profit_diff_optimal:,.2f} a mais!")
        else:
            print(
                f"\n  Modelo NÃO VIÁVEL - Melhor threshold ainda perde R$ {abs(profit_diff_optimal):,.2f}")
            print(
                f"    Recomendação: Usar modelo para timing/personalização, não seleção")

        return {
            'optimal_profit': optimal_profit.to_dict(),
            'optimal_roi': optimal_roi.to_dict(),
            'optimal_f1': optimal_f1.to_dict(),
            'baseline_profit': baseline_profit,
            'all_results': df_results,
            'recommended_threshold': optimal_profit['threshold'],
            'viable': profit_diff_optimal > 0
        }

    def plot_threshold_optimization(
        self,
        y_pred_proba: np.ndarray,
        cost_per_contact: float = 5.0,
        revenue_per_conversion: float = 200.0,
        save: bool = False
    ):
        """
        Plota análise de threshold para maximizar lucro.
        """
        print("\n" + "="*70)
        print("VISUALIZAÇÃO: OTIMIZAÇÃO DE THRESHOLD")
        print("="*70 + "\n")

        # Obter resultados da otimização
        opt_results = self.optimize_threshold_for_profit(
            y_pred_proba,
            cost_per_contact,
            revenue_per_conversion
        )

        df_results = opt_results['all_results']
        baseline_profit = opt_results['baseline_profit']

        thresholds = df_results['threshold'].values
        profits = df_results['profit'].values
        rois = df_results['roi'].values
        precisions = df_results['precision'].values
        recalls = df_results['recall'].values
        f1s = df_results['f1'].values
        contacted_pcts = df_results['contact_rate'].values * 100

        # Plotar
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Lucro vs Threshold
        ax = axes[0, 0]
        ax.plot(thresholds, profits, linewidth=3,
                marker='o', markersize=3, color='#06A77D')
        ax.axhline(baseline_profit, color='red', linestyle='--', linewidth=2,
                   label=f'Baseline: R$ {baseline_profit:,.0f}')

        optimal_threshold = opt_results['optimal_profit']['threshold']
        max_profit = opt_results['optimal_profit']['profit']
        ax.axvline(optimal_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Ótimo: {optimal_threshold:.2f}')
        ax.scatter(optimal_threshold, max_profit, color='red', s=200, zorder=5)

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Lucro (R$)', fontsize=12)
        ax.set_title('Lucro vs Threshold', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'R$ {x/1000:.0f}k'))

        # 2. ROI vs Threshold
        ax = axes[0, 1]
        ax.plot(thresholds, rois, linewidth=3,
                marker='o', markersize=3, color='#2E86AB')

        baseline_roi = opt_results['baseline_profit'] / \
            (self.total_population * cost_per_contact) * 100
        ax.axhline(baseline_roi, color='red', linestyle='--', linewidth=2,
                   label=f'Baseline: {baseline_roi:.0f}%')

        optimal_roi_threshold = opt_results['optimal_roi']['threshold']
        ax.axvline(optimal_roi_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Ótimo: {optimal_roi_threshold:.2f}')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('ROI (%)', fontsize=12)
        ax.set_title('ROI vs Threshold', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Precision, Recall, F1 vs Threshold
        ax = axes[1, 0]
        ax.plot(thresholds, precisions, linewidth=2,
                marker='o', markersize=2, label='Precision', color='#A23B72')
        ax.plot(thresholds, recalls, linewidth=2,
                marker='s', markersize=2, label='Recall', color='#F18F01')
        ax.plot(thresholds, f1s, linewidth=2, marker='^', markersize=2,
                label='F1-Score', color='#06A77D')
        ax.axvline(0.5, color='gray', linestyle=':',
                   linewidth=2, label='Default (0.5)')
        ax.axvline(optimal_threshold, color='orange', linestyle='--',
                   linewidth=2, label=f'Ótimo Lucro ({optimal_threshold:.2f})')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Métricas de Classificação vs Threshold',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. % Clientes Contatados vs Threshold
        ax = axes[1, 1]
        ax.plot(thresholds, contacted_pcts, linewidth=3,
                marker='o', markersize=3, color='#C73E1D')
        ax.axvline(optimal_threshold, color='orange', linestyle='--',
                   linewidth=2, label=f'Ótimo: {optimal_threshold:.2f}')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('% População Contatada', fontsize=12)
        ax.set_title('Volume de Contatos vs Threshold',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('threshold_optimization.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print(
            f"\n Threshold ótimo para LUCRO: {optimal_threshold:.2f} (Lucro: R$ {max_profit:,.2f})")
        print(
            f" Variação vs baseline: R$ {max_profit - baseline_profit:+,.2f}")
        print()

    # ========================================================================
    # 6. PIPELINE COMPLETO
    # ========================================================================

    def run_complete_pipeline(
        self,
        tune_model: str = 'XGBoost',
        save_plots: bool = True,
        cost_per_contact: float = 5.0,
        revenue_per_conversion: float = 200.0
    ) -> Dict:
        """
        Executa pipeline completo de modelagem.

        Args:
            tune_model: modelo a otimizar
            save_plots: salvar todas as figuras
            cost_per_contact: custo por ligação (para business metrics)
            revenue_per_conversion: receita por conversão (para business metrics)

        Returns:
            Dicionário com resultados finais
        """
        print("\n" + "="*70)
        print("PIPELINE COMPLETO DE MODELAGEM - BANK MARKETING")
        print("="*70 + "\n")

        # 1. Baseline Models
        print("ETAPA 1/6: BASELINE MODELS")
        print("="*70)
        results_df = self.train_baseline_models()
        self.plot_baseline_comparison(save=save_plots)
        self.plot_confusion_matrices(save=save_plots)
        self.plot_roc_curves(save=save_plots)

        # 2. Feature Selection
        print("\nETAPA 2/6: FEATURE SELECTION")
        print("="*70)
        feature_scores = self.feature_selection_analysis(k_features=20)
        self.plot_feature_importance(top_n=20, save=save_plots)

        # 3. Hyperparameter Tuning
        print("\nETAPA 3/6: HYPERPARAMETER TUNING")
        print("="*70)
        tuned_model = self.hyperparameter_tuning(
            model_name=tune_model,
            method='random',
            n_iter=100,
            cv=5
        )

        # 4. Interpretabilidade
        print("\nETAPA 4/6: INTERPRETABILIDADE (SHAP)")
        print("="*70)
        self.explain_model_shap(
            tuned_model['model'], max_display=20, save=save_plots)

        # 5. Threshold Optimization & Business Metrics
        print("\nETAPA 5/6: BUSINESS METRICS & THRESHOLD OPTIMIZATION")
        print("="*70)

        self.plot_threshold_optimization(
            tuned_model['y_pred_proba'],
            cost_per_contact=cost_per_contact,
            revenue_per_conversion=revenue_per_conversion,
            save=save_plots
        )

        optimization_results = self.optimize_threshold_for_profit(
            tuned_model['y_pred_proba'],
            cost_per_contact=cost_per_contact,
            revenue_per_conversion=revenue_per_conversion
        )

        optimal_threshold = optimization_results['recommended_threshold']

        business_metrics = self.calculate_business_metrics(
            tuned_model['y_pred_proba'],
            cost_per_contact=cost_per_contact,
            revenue_per_conversion=revenue_per_conversion,
            threshold=optimal_threshold
        )

        # 6. Relatório Final
        print("\nETAPA 6/6: RELATÓRIO FINAL")
        print("="*70)
        print("\nRESULTADOS FINAIS DO PIPELINE")
        print("="*70)

        print("\nMELHOR MODELO:")
        print(f"   Algoritmo: {tune_model}")
        print(f"   F1-Score (CV): {tuned_model['cv_score']:.4f}")
        print(f"   F1-Score (Test): {tuned_model['test_f1']:.4f}")
        print(f"   AUC-ROC (Test): {tuned_model['test_auc']:.4f}")

        print("\nIMPACTO DE NEGÓCIO:")
        print(f"   Threshold ótimo: {optimal_threshold:.2f}")
        print(f"   Modelo é viável? {business_metrics['impact']['viable']}")
        if business_metrics['impact']['viable']:
            print(
                f"    Aumento de lucro: R$ {business_metrics['impact']['profit_increase']:,.2f}")
            print(
                f"    ROI do modelo: {business_metrics['model']['roi']:.2f}%")
            print(
                f"    Redução de custos: {business_metrics['impact']['cost_reduction_pct']:.1f}%")
        else:
            print(
                f"     Redução de lucro: R$ {abs(business_metrics['impact']['profit_increase']):,.2f}")
            print(
                f"    Modelo útil para: timing, personalização, ou quando custo > R$ 12/contato")

        print("\nTOP 5 FEATURES MAIS IMPORTANTES:")
        top_features = feature_scores.head(5)['feature'].tolist()
        for i, feat in enumerate(top_features, 1):
            # Marcar features V2
            marker = "Features 'novas'" if feat in ['contact_history_weighted', 'campaign_density',
                                                    'timing_score', 'contact_efficiency',
                                                    'financial_health_score', 'high_balance_prev_success'] else ""
            print(f"   {i}. {feat}{marker}")

        print("\n" + "="*70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!!!!!!!")
        print("="*70 + "\n")

        return {
            'baseline_results': results_df,
            'feature_importance': feature_scores,
            'tuned_model': tuned_model,
            'optimal_threshold': optimal_threshold,
            'business_metrics': business_metrics,
            'optimization_results': optimization_results,
            'model_viable': business_metrics['impact']['viable']
        }
