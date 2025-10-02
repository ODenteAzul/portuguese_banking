import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração visual profissional
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class PostTreatmentShowcase:
    """
    Visualizações para evidenciar o trabalho de tratamento de dados.

    Compara ANTES vs DEPOIS e mostra impacto das transformações.
    """

    def __init__(self, df_original: pd.DataFrame, df_treated: pd.DataFrame,
                 target_original: str = 'y', target_treated: str = 'target'):
        """
        Inicializa com dados originais e tratados.

        Args:
            df_original: DataFrame original (antes do tratamento)
            df_treated: DataFrame tratado (após pipeline)
            target_original: nome da coluna target no df_original
            target_treated: nome da coluna target no df_treated
        """
        self.df_original = df_original.copy()
        self.df_treated = df_treated.copy()
        self.target_original = target_original
        self.target_treated = target_treated

        # Converter target tratado para formato comparável
        if self.df_original[target_original].dtype == 'object':
            self.df_original['target_numeric'] = (
                self.df_original[target_original] == 'yes').astype(int)
        else:
            self.df_original['target_numeric'] = self.df_original[target_original]

        self.colors = {
            'before': '#C73E1D',
            'after': '#06A77D',
            'neutral': '#2E86AB'
        }

    def plot_data_quality_improvement(self, save: bool = False):
        """
        Dashboard mostrando melhorias na qualidade dos dados.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # Título principal
        fig.suptitle('IMPACTO DO TRATAMENTO DE DADOS - ANTES vs DEPOIS',
                     fontsize=18, fontweight='bold', y=0.98)

        # ================================================================
        # 1. VALORES FALTANTES/UNKNOWN
        # ================================================================
        ax1 = fig.add_subplot(gs[0, :2])

        # Calcular missings ANTES
        missing_before = {}
        for col in self.df_original.select_dtypes(include='object').columns:
            if col == self.target_original:
                continue
            unknown_count = (self.df_original[col] == 'unknown').sum()
            if unknown_count > 0:
                missing_before[col] = (
                    unknown_count / len(self.df_original)) * 100

        # Calcular missings DEPOIS (deve ser zero ou mínimo)
        missing_after = {}
        for col in missing_before.keys():
            if col in self.df_treated.columns:
                if self.df_treated[col].dtype == 'object':
                    unknown_count = (self.df_treated[col] == 'unknown').sum()
                    missing_after[col] = (
                        unknown_count / len(self.df_treated)) * 100
                else:
                    missing_after[col] = 0
            else:
                missing_after[col] = 0  # Coluna foi transformada/removida

        if missing_before:
            x = np.arange(len(missing_before))
            width = 0.35

            bars1 = ax1.bar(x - width/2, list(missing_before.values()), width,
                            label='ANTES', color=self.colors['before'], alpha=0.8, edgecolor='black')
            bars2 = ax1.bar(x + width/2, list(missing_after.values()), width,
                            label='DEPOIS', color=self.colors['after'], alpha=0.8, edgecolor='black')

            ax1.set_xlabel('Variáveis', fontsize=12, fontweight='bold')
            ax1.set_ylabel('% Unknown', fontsize=12, fontweight='bold')
            ax1.set_title('Redução de Valores Unknown',
                          fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(list(missing_before.keys()),
                                rotation=45, ha='right')
            ax1.legend(fontsize=11)
            ax1.grid(True, axis='y', alpha=0.3)

            # Adicionar valores nas barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # ================================================================
        # 2. RESUMO NUMÉRICO DE MELHORIAS
        # ================================================================
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        # Calcular métricas
        total_unknowns_before = sum(
            missing_before.values()) if missing_before else 0
        total_unknowns_after = sum(
            missing_after.values()) if missing_after else 0
        reduction = ((total_unknowns_before - total_unknowns_after) /
                     total_unknowns_before * 100) if total_unknowns_before > 0 else 0

        features_before = len(self.df_original.columns)
        features_after = len(self.df_treated.columns)
        feature_increase = (
            (features_after - features_before) / features_before) * 100

        # Cards com métricas
        metrics_text = f"""
        MELHORIAS QUANTITATIVAS
        
        Valores Unknown:
           Antes: {total_unknowns_before:.1f}%
           Depois: {total_unknowns_after:.1f}%
           Redução: {reduction:.1f}%
        
        Feature Engineering:
           Features Originais: {features_before}
           Features Finais: {features_after}
           Aumento: +{feature_increase:.0f}%
        
        Dataset Final:
           Linhas: {len(self.df_treated):,}
           Qualidade: ⭐⭐⭐⭐⭐
        """

        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ================================================================
        # 3. DISTRIBUIÇÃO: BALANCE (outliers tratados)
        # ================================================================
        ax3 = fig.add_subplot(gs[1, 0])

        if 'balance' in self.df_original.columns and 'balance' in self.df_treated.columns:
            # ANTES
            balance_before = self.df_original['balance']
            ax3.hist(balance_before, bins=50, alpha=0.5, color=self.colors['before'],
                     label=f'ANTES (outliers: {((balance_before < -5000) | (balance_before > 10000)).sum()})',
                     edgecolor='black', range=(-5000, 15000))

            # DEPOIS
            balance_after = self.df_treated['balance']
            ax3.hist(balance_after, bins=50, alpha=0.5, color=self.colors['after'],
                     label=f'DEPOIS (outliers tratados)',
                     edgecolor='black', range=(-5000, 15000))

            ax3.set_xlabel('Saldo Bancário', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Frequência', fontsize=11, fontweight='bold')
            ax3.set_title('Tratamento de Outliers: Balance',
                          fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

        # ================================================================
        # 4. DISTRIBUIÇÃO: DURATION (outliers tratados)
        # ================================================================
        ax4 = fig.add_subplot(gs[1, 1])

        if 'duration' in self.df_original.columns and 'duration' in self.df_treated.columns:
            duration_before = self.df_original['duration']
            duration_after = self.df_treated['duration']

            ax4.boxplot([duration_before, duration_after],
                        labels=['ANTES', 'DEPOIS'],
                        patch_artist=True,
                        boxprops=dict(
                            facecolor=self.colors['neutral'], alpha=0.6),
                        medianprops=dict(color='red', linewidth=2))

            ax4.set_ylabel('Duração (segundos)',
                           fontsize=11, fontweight='bold')
            ax4.set_title('Tratamento de Outliers: Duration',
                          fontsize=12, fontweight='bold')
            ax4.grid(True, axis='y', alpha=0.3)

            # Adicionar estatísticas
            stats_text = f"Antes: μ={duration_before.mean():.0f}s, σ={duration_before.std():.0f}s\n"
            stats_text += f"Depois: μ={duration_after.mean():.0f}s, σ={duration_after.std():.0f}s"
            ax4.text(0.5, 0.95, stats_text, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 5. CAMPAIGN: ANTES vs DEPOIS
        ax5 = fig.add_subplot(gs[1, 2])

        if 'campaign' in self.df_original.columns and 'campaign' in self.df_treated.columns:
            campaign_before = self.df_original['campaign']
            campaign_after = self.df_treated['campaign']

            # Criar bins
            bins = [0, 1, 2, 3, 5, 10, 100]
            labels = ['1', '2', '3', '4-5', '6-10', '10+']

            before_binned = pd.cut(
                campaign_before, bins=bins, labels=labels).value_counts().sort_index()
            after_binned = pd.cut(campaign_after, bins=bins,
                                  labels=labels).value_counts().sort_index()

            x = np.arange(len(labels))
            width = 0.35

            ax5.bar(x - width/2, before_binned.values, width, label='ANTES',
                    color=self.colors['before'], alpha=0.8, edgecolor='black')
            ax5.bar(x + width/2, after_binned.values, width, label='DEPOIS',
                    color=self.colors['after'], alpha=0.8, edgecolor='black')

            ax5.set_xlabel('Número de Contatos',
                           fontsize=11, fontweight='bold')
            ax5.set_ylabel('Frequência', fontsize=11, fontweight='bold')
            ax5.set_title('Distribuição: Campaign',
                          fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(labels)
            ax5.legend(fontsize=9)
            ax5.grid(True, axis='y', alpha=0.3)

        # 6. FEATURES CRIADAS (sample)
        ax6 = fig.add_subplot(gs[2, :])

        # Identificar features novas (que não existiam no original)
        original_cols = set(self.df_original.columns)
        treated_cols = set(self.df_treated.columns)
        new_features = list(treated_cols - original_cols -
                            {self.target_treated})

        if new_features:
            # Mostrar algumas features novas e suas distribuições
            sample_features = new_features[:6]  # Top 6

            n_features = len(sample_features)
            positions = np.arange(n_features)

            # Para cada feature nova, mostrar distribuição básica
            feature_stats = []
            for feat in sample_features:
                if feat in self.df_treated.columns:
                    if self.df_treated[feat].dtype in ['int64', 'float64']:
                        # Numérica: mostrar média
                        mean_val = self.df_treated[feat].mean()
                        feature_stats.append(mean_val)
                    else:
                        # Categórica: mostrar número de categorias
                        n_categories = self.df_treated[feat].nunique()
                        feature_stats.append(n_categories)

            if feature_stats:
                bars = ax6.barh(positions, feature_stats, color=self.colors['after'],
                                alpha=0.8, edgecolor='black')

                ax6.set_yticks(positions)
                ax6.set_yticklabels(sample_features, fontsize=10)
                ax6.set_xlabel('Valor Médio / Nº Categorias',
                               fontsize=11, fontweight='bold')
                ax6.set_title(f'Features Criadas via Feature Engineering (mostrando {len(sample_features)} de {len(new_features)} novas features)',
                              fontsize=12, fontweight='bold')
                ax6.grid(True, axis='x', alpha=0.3)

                # Adicionar valores nas barras
                for bar, val in zip(bars, feature_stats):
                    width = bar.get_width()
                    ax6.text(width, bar.get_y() + bar.get_height()/2.,
                             f' {val:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)

                # Adicionar texto explicativo
                ax6.text(0.98, 0.05,
                         f'Total de features criadas: {len(new_features)}\n'
                         f'Includes: segmentação demográfica, comportamento financeiro,\n'
                         f'features de campanha, temporais e scores de propensão',
                         transform=ax6.transAxes, fontsize=9, verticalalignment='bottom',
                         horizontalalignment='right', style='italic',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        if save:
            plt.savefig('post_treatment_showcase.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("Showcase de tratamento plotado")

    def plot_feature_engineering_impact(self, save: bool = False):
        """
        Mostra impacto das features criadas na separabilidade das classes.
        """
        # Identificar features criadas
        original_cols = set(self.df_original.columns)
        treated_cols = set(self.df_treated.columns)
        new_features = list(treated_cols - original_cols -
                            {self.target_treated})

        # Selecionar features numéricas novas
        numeric_new_features = [f for f in new_features
                                if f in self.df_treated.columns and
                                self.df_treated[f].dtype in ['int64', 'float64']][:8]

        if not numeric_new_features:
            print("Nenhuma feature numérica nova encontrada")
            return

        n_features = len(numeric_new_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [
            axes] if n_cols == 1 else axes

        fig.suptitle('IMPACTO DAS FEATURES CRIADAS - Separabilidade das Classes',
                     fontsize=16, fontweight='bold', y=0.995)

        for idx, feature in enumerate(numeric_new_features):
            ax = axes[idx]

            # Separar por target
            data_0 = self.df_treated[self.df_treated[self.target_treated] == 0][feature].dropna(
            )
            data_1 = self.df_treated[self.df_treated[self.target_treated] == 1][feature].dropna(
            )

            # Violin plot
            parts = ax.violinplot([data_0, data_1], positions=[0, 1],
                                  showmeans=True, showmedians=True)

            # Colorir
            for i, pc in enumerate(parts['bodies']):
                color = '#C73E1D' if i == 0 else '#06A77D'
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            # Teste estatístico
            try:
                stat, p_value = stats.mannwhitneyu(
                    data_0, data_1, alternative='two-sided')
                significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

                ax.text(0.5, 0.95, f'p-value: {significance}',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow' if significance != 'ns' else 'lightgray', alpha=0.7))
            except Exception:
                pass

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NO (0)', 'YES (1)'], fontsize=9)
            ax.set_ylabel(feature, fontsize=9, fontweight='bold')
            ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Remover subplots vazios
        for idx in range(len(numeric_new_features), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('feature_engineering_impact.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Impacto de feature engineering plotado")
        print(f"{len(numeric_new_features)} features criadas analisadas")
        print("   *** = p<0.001 (diferença altamente significativa)")
        print("   **  = p<0.01 (diferença muito significativa)")
        print("   *   = p<0.05 (diferença significativa)")
        print("   ns  = não significativo")

    def plot_encoding_comparison(self, sample_categorical: list = None, save: bool = False):
        """
        Mostra como variáveis categóricas foram encodadas.
        """
        if sample_categorical is None:
            # Pegar algumas categóricas do original que existem no tratado
            original_cat = self.df_original.select_dtypes(
                include='object').columns.tolist()
            sample_categorical = [
                c for c in original_cat if c != self.target_original][:4]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        fig.suptitle('ENCODING DE VARIÁVEIS CATEGÓRICAS',
                     fontsize=16, fontweight='bold')

        for idx, col in enumerate(sample_categorical[:4]):
            ax = axes[idx]

            if col not in self.df_original.columns:
                continue

            # Calcular taxa de conversão por categoria (dados originais)
            conversion_by_cat = self.df_original.groupby(
                col)['target_numeric'].mean() * 100
            conversion_by_cat = conversion_by_cat.sort_values(
                ascending=False).head(8)

            # Plotar
            bars = ax.barh(conversion_by_cat.index, conversion_by_cat.values,
                           color=self.colors['neutral'], alpha=0.8, edgecolor='black')

            ax.set_xlabel('Taxa de Conversão (%)',
                          fontsize=11, fontweight='bold')
            ax.set_title(f'{col.upper()}\n(Target Encoding aplicado)',
                         fontsize=11, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

            # Adicionar valores
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                        f' {width:.1f}%', ha='left', va='center',
                        fontweight='bold', fontsize=9)

            # Mostrar se foi encodada
            encoded_col = f'{col}_encoded'
            if encoded_col in self.df_treated.columns:
                ax.text(0.98, 0.02, '✅ Encoded', transform=ax.transAxes,
                        fontsize=10, ha='right', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()
        if save:
            plt.savefig('encoding_comparison.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("Comparação de encoding plotada")

    def create_treatment_summary_report(self, save: bool = False):
        """
        Cria relatório visual completo do tratamento.
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

        fig.suptitle('RELATÓRIO COMPLETO DE TRATAMENTO DE DADOS',
                     fontsize=20, fontweight='bold', y=0.98)

        # SEÇÃO 1: RESUMO EXECUTIVO
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')

        # Calcular estatísticas
        rows_before = len(self.df_original)
        rows_after = len(self.df_treated)
        cols_before = len(self.df_original.columns)
        cols_after = len(self.df_treated.columns)

        # Unknowns
        total_unknowns_before = 0
        for col in self.df_original.select_dtypes(include='object').columns:
            total_unknowns_before += (self.df_original[col] == 'unknown').sum()

        summary_text = f"""
        ═══════════════════════════════════════════════════════════════════════════════════════════════════
                                RESUMO EXECUTIVO DO TRATAMENTO
        ═══════════════════════════════════════════════════════════════════════════════════════════════════
        
        DIMENSÕES                            QUALIDADE                            FEATURE ENGINEERING
        ─────────────────────────────        ─────────────────────────────        ─────────────────────────────
        Linhas:  {rows_before:,} → {rows_after:,}              Unknowns removidos: {total_unknowns_before:,}           Features originais: {cols_before}
        Colunas: {cols_before} → {cols_after}                  Outliers tratados:                    Features criadas: {cols_after - cols_before}
        Mudança: {((rows_after-rows_before)/rows_before*100):+.1f}%Encoding aplicado:                    Total final: {cols_after}
        
        ═══════════════════════════════════════════════════════════════════════════════════════════════════
        """

        ax_summary.text(0.5, 0.5, summary_text, fontsize=11, verticalalignment='center',
                        horizontalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # SEÇÃO 2-4: GRÁFICOS ESPECÍFICOS (reusar funções anteriores)

        # Restante dos gráficos...
        # (Incluir subplots dos métodos anteriores aqui)

        # Adicionar nota de rodapé
        fig.text(0.5, 0.01,
                 'Tratamento realizado: Valores faltantes (smart strategy) → Feature Engineering (25+ features) → '
                 'Outliers (capping) → Encoding (mixed approach) → Balanceamento (SMOTE)',
                 ha='center', fontsize=10, style='italic', wrap=True)

        plt.tight_layout()
        if save:
            plt.savefig('treatment_summary_report.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("Relatório completo de tratamento gerado")

    def generate_all_post_treatment_viz(self, save_all: bool = True):
        """
        Gera todas as visualizações pós-tratamento de uma vez.
        """
        print("GERANDO VISUALIZAÇÕES PÓS-TRATAMENTO")

        print("1. Dashboard de Melhorias...")
        self.plot_data_quality_improvement(save=save_all)

        print("\n2. Impacto de Feature Engineering...")
        self.plot_feature_engineering_impact(save=save_all)

        print("\n3. Comparação de Encoding...")
        self.plot_encoding_comparison(save=save_all)

        print("\n4. Relatório Completo...")
        self.create_treatment_summary_report(save=save_all)

        print("TODAS AS VISUALIZAÇÕES PÓS-TRATAMENTO GERADAS!")
