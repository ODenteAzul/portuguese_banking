import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple, Optional
import warnings
from matplotlib.colors import to_rgba
warnings.filterwarnings('ignore')

# Configuração de estilo profissional
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11


class BankDataVisualization:
    """
    Classe para visualizações profissionais do Bank Marketing Dataset.

    Organizada em 5 categorias:
    1. Análise Univariada (distribuições)
    2. Análise Bivariada (relação com target)
    3. Correlações e Multicolinearidade
    4. Segmentação de Clientes
    5. Insights de Campanha
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'y'):
        """
        Inicializa com DataFrame e configura paleta de cores.

        Args:
            df: DataFrame original
            target_col: nome da coluna target
        """
        self.df = df.copy()
        self.target_col = target_col

        # Paleta de cores profissional
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'yes': '#06A77D',
            'no': '#C73E1D'
        }

        # Separar colunas por tipo
        self.numeric_cols = self.df.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=['object']).columns.tolist()

        if self.target_col in self.categorical_cols:
            self.categorical_cols.remove(self.target_col)

    # antes eu criava o grid em cada método, agora unifiquei

    def _criar_grid(self, cols: List[str], n_cols: int = 2, row_height: int = 5):
        """
        Cria um grid de subplots consistente para os gráficos.

        Args:
            cols: lista de colunas que serão plotadas
            n_cols: número de colunas fixas no grid (default=2)
            row_height: altura por linha para escalonar figura

        Returns:
            fig, axes (já flatten)
        """
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 7, n_rows * row_height)
        )
        axes = axes.flatten() if n_rows > 1 else [axes]
        return fig, axes

    # 1 ANÁLISE UNIVARIADA

    def plot_distribuicao_numericas(self, cols: Optional[List[str]] = None, save: bool = False):
        """
        Plota distribuições das variáveis numéricas com histograma + KDE + boxplot.

        Args:
            cols: lista de colunas (None = todas)
            save: salvar figura
        """

        if cols is None:
            cols = self.numeric_cols

        fig, axes = self._criar_grid(cols, n_cols=2)

        for idx, col in enumerate(cols):
            ax_hist = axes[idx]

            # Histograma + KDE
            data = self.df[col].dropna()
            ax_hist.hist(data,
                         bins=50,
                         alpha=0.6,
                         color=self.colors['primary'],
                         edgecolor='black')

            # KDE
            ax_kde = ax_hist.twinx()

            data.plot.kde(
                ax=ax_kde,
                color=self.colors['secondary'],
                linewidth=2,
                alpha=0.9)

            ax_kde.set_ylabel('Densidade',
                              color=self.colors['secondary'])

            ax_kde.tick_params(axis='y',
                               labelcolor=self.colors['secondary'])

            # Estatísticas
            mean_val = data.mean()
            median_val = data.median()
            ax_hist.axvline(mean_val,
                            color='red',
                            alpha=0.8,
                            linestyle='--',
                            linewidth=2,
                            label=f'Média: {mean_val:.1f}')
            ax_hist.axvline(median_val,
                            color='green',
                            alpha=0.8,
                            linestyle='--',
                            linewidth=2,
                            label=f'Mediana: {median_val:.1f}')

            ax_hist.set_title(
                f'Distribuição: {col}',
                fontsize=12,
                fontweight='bold')

            ax_hist.set_xlabel(col)
            ax_hist.set_ylabel('Frequência')
            ax_hist.legend(loc='upper right')
            ax_hist.grid(True, alpha=0.3)

        # Remover subplots vazios
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('dist_numericas.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"{len(cols)} distribuições numéricas plotadas")

    def plot_distribuicao_categoricas(self, cols: Optional[List[str]] = None, top_n: int = 10, save: bool = False):
        """
        Plota distribuições das variáveis categóricas com barras horizontais.

        Args:
            cols: lista de colunas (None = todas)
            top_n: mostrar top N categorias
            save: salvar figura
        """
        if cols is None:
            cols = self.categorical_cols

        n_cols = len(cols)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]

            # Pegar top N categorias
            value_counts = self.df[col].value_counts().head(top_n)
            total = len(self.df)
            percentages = (value_counts / total * 100).round(2)

            # Criar barras horizontais
            y_pos = np.arange(len(value_counts))
            bars = ax.barh(y_pos, value_counts.values,
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')

            # Adicionar percentuais nas barras
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                        f' {pct}%', ha='left', va='center', fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(value_counts.index)
            ax.set_xlabel('Frequência')
            ax.set_title(f'{col.upper()} (Top {min(top_n, len(value_counts))})',
                         fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

        # Remover subplots vazios
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('dist_categoricas.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ {len(cols)} distribuições categóricas plotadas")

    # ANÁLISE BIVARIADA (RELAÇÃO COM TARGET)

    def plot_numericas_vs_target(self, cols: Optional[List[str]] = None, save: bool = False):
        """
        Compara distribuições numéricas entre classes do target (boxplot + violin).

        Args:
            cols: lista de colunas (None = todas)
            save: salvar figura
        """
        if cols is None:
            cols = self.numeric_cols

        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]

            # Violin plot com overlay de boxplot
            data_to_plot = [
                self.df[self.df[self.target_col] == 'no'][col].dropna(),
                self.df[self.df[self.target_col] == 'yes'][col].dropna()
            ]

            parts = ax.violinplot(data_to_plot, positions=[
                                  0, 1], showmeans=True, showmedians=True)

            # Colorir violins
            for i, pc in enumerate(parts['bodies']):
                color = self.colors['no'] if i == 0 else self.colors['yes']
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            # Boxplot overlay
            bp = ax.boxplot(data_to_plot, positions=[0, 1], widths=0.15, patch_artist=True,
                            boxprops=dict(facecolor='white', alpha=0.8),
                            medianprops=dict(color='black', linewidth=2))

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NO', 'YES'])
            ax.set_ylabel(col)
            ax.set_title(f'{col} vs Target', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Teste estatístico (Mann-Whitney U)
            stat, p_value = stats.mannwhitneyu(
                data_to_plot[0], data_to_plot[1], alternative='two-sided')
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            ax.text(0.5, 0.95, f'p-value: {significance}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Remover subplots vazios
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('numericas_vs_target.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ {len(cols)} comparações numéricas vs target plotadas")

    def plot_categoricas_vs_target(self, cols: Optional[List[str]] = None, save: bool = False):
        """
        Mostra taxa de conversão por categoria (stacked bars + taxa).

        Args:
            cols: lista de colunas (None = todas)
            save: salvar figura
        """
        if cols is None:
            cols = self.categorical_cols[:6]  # Top 6 para não sobrecarregar

        n_cols = len(cols)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 5))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]

            # Calcular proporções
            ct = pd.crosstab(
                self.df[col], self.df[self.target_col], normalize='index') * 100
            # Ordenar por taxa de conversão
            ct = ct.sort_values('yes', ascending=True)

            # Stacked bar horizontal
            ct.plot(kind='barh', stacked=True, ax=ax,
                    color=[self.colors['no'], self.colors['yes']],
                    alpha=0.8, edgecolor='black')

            # Adicionar taxa de conversão no final das barras
            for i, (idx_val, row) in enumerate(ct.iterrows()):
                conversion_rate = row['yes']
                ax.text(102,
                        i,
                        f'{conversion_rate:.1f}%',
                        va='center',
                        fontweight='bold',
                        fontsize=9)

            ax.set_xlabel('Percentual (%)')
            ax.set_title(
                f'Taxa de Conversão por {col.upper()}',
                fontsize=12, fontweight='bold')
            ax.legend(title='Target',
                      labels=['NO', 'YES'])
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_xlim(0, 110)

        # Remover subplots vazios
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save:
            plt.savefig('categoricas_vs_target.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ {len(cols)} análises categóricas vs target plotadas")

    # CORRELAÇÕES E MULTICOLINEARIDADE

    def plot_correlation_matrix(self,
                                method: str = 'pearson',
                                save: bool = False):
        """
        Plota matriz de correlação com heatmap.

        Args:
            method: 'pearson' ou 'spearman'
            save: salvar figura
        """
        # Calcular correlação apenas para numéricas
        corr_matrix = self.df[self.numeric_cols].corr(method=method)

        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig, ax = plt.subplots(figsize=(14, 12))

        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    ax=ax)

        ax.set_title(f'Matriz de Correlação ({method.capitalize()})',
                     fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        if save:
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Identificar correlações fortes (|r| > 0.7)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        if high_corr:
            print("\n  Correlações fortes detectadas (|r| > 0.7):")
            for item in high_corr:
                print(
                    f"""   {item['var1']} ↔ {item['var2']}:
                    {item['correlation']:.3f}""")
            print("   → Considerar remover uma das variáveis ou usar PCA\n")
        else:
            print("\n✓ Nenhuma correlação forte detectada (|r| > 0.7)\n")

    def plot_vif_analysis(self, save: bool = False):
        """
        Calcula e plota VIF (Variance Inflation Factor) para
        detectar multicolinearidade.

        VIF > 10: multicolinearidade severa
        VIF > 5: multicolinearidade moderada
        """

        # Preparar dados (apenas numéricas, sem NaN)
        df_numeric = self.df[self.numeric_cols].dropna()

        # Calcular VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i)
                           for i in range(len(df_numeric.columns))]

        vif_data = vif_data.sort_values('VIF', ascending=False)

        # Plotar
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [self.colors['danger'] if vif > 10 else
                  self.colors['warning'] if vif > 5 else
                  self.colors['success'] for vif in vif_data['VIF']]

        bars = ax.barh(vif_data['Feature'], vif_data['VIF'],
                       color=colors, alpha=0.8, edgecolor='black')

        # Linhas de referência
        ax.axvline(5, color='orange', linestyle='--',
                   linewidth=2, label='VIF = 5 (moderado)')
        ax.axvline(10, color='red', linestyle='--',
                   linewidth=2, label='VIF = 10 (severo)')

        ax.set_xlabel('VIF (Variance Inflation Factor)', fontsize=12)
        ax.set_title('Análise de Multicolinearidade (VIF)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('vif_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Alertas
        high_vif = vif_data[vif_data['VIF'] > 10]
        if not high_vif.empty:
            print("\n  Multicolinearidade SEVERA detectada (VIF > 10):")
            for _, row in high_vif.iterrows():
                print(f"   {row['Feature']}: VIF = {row['VIF']:.2f}")
            print("   → Remover ou combinar variáveis correlacionadas\n")
        else:
            print("\n✓ Multicolinearidade sob controle (todos VIF < 10)\n")

        return vif_data

    # SEGMENTAÇÃO DE CLIENTES

    def plot_segmentacao_demografica(self, save: bool = False):
        """
        Análise de segmentação demográfica:
        idade × profissão × educação vs conversão.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Idade vs Target
        ax = axes[0, 0]
        age_bins = [18, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        self.df['age_group'] = pd.cut(
            self.df['age'], bins=age_bins, labels=age_labels)

        conversion_by_age = self.df.groupby('age_group')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax.bar(conversion_by_age.index,
                      conversion_by_age.values,
                      color=self.colors['primary'],
                      alpha=0.8,
                      edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Faixa Etária',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontweight='bold')

        # 2. Profissão vs Target (Top 8)
        ax = axes[0, 1]
        conversion_by_job = self.df.groupby('job')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False).head(8)

        bars = ax.barh(conversion_by_job.index,
                       conversion_by_job.values,
                       color=self.colors['secondary'],
                       alpha=0.8,
                       edgecolor='black')
        ax.set_xlabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Profissão (Top 8)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {width:.1f}%', ha='left', va='center', fontweight='bold')

        # 3. Educação vs Target
        ax = axes[1, 0]
        conversion_by_edu = self.df.groupby('education')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False)

        bars = ax.bar(conversion_by_edu.index,
                      conversion_by_edu.values,
                      color=self.colors['success'],
                      alpha=0.8,
                      edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Nível Educacional',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Estado Civil vs Target
        ax = axes[1, 1]
        conversion_by_marital = self.df.groupby('marital')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False)

        bars = ax.bar(conversion_by_marital.index,
                      conversion_by_marital.values,
                      color=self.colors['warning'],
                      alpha=0.8,
                      edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Estado Civil',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        if save:
            plt.savefig('segmentacao_demografica.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Segmentação demográfica plotada")

    def plot_perfil_financeiro(self, save: bool = False):
        """
        Análise do perfil financeiro: saldo, empréstimos, inadimplência.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distribuição de Saldo por Target
        ax = axes[0, 0]

        for target_val in ['no', 'yes']:
            data = self.df[self.df[self.target_col] == target_val]['balance']
            ax.hist(data, bins=50, alpha=0.6,
                    label=target_val.upper(),
                    color=self.colors[target_val])

        ax.set_xlabel('Saldo Bancário', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.set_title('Distribuição de Saldo por Target',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5000, 15000)  # Zoom na região de interesse

        # 2. Taxa de Conversão por Categoria de Saldo
        ax = axes[0, 1]

        balance_bins = [-np.inf, 0, 500, 2000, 5000, np.inf]
        balance_labels = ['Negativo', '0-500', '501-2k', '2k-5k', '5k+']
        self.df['balance_cat'] = pd.cut(
            self.df['balance'], bins=balance_bins, labels=balance_labels)

        conversion_by_balance = self.df.groupby('balance_cat')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax.bar(conversion_by_balance.index, conversion_by_balance.values,
                      color=self.colors['primary'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Faixa de Saldo',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Empréstimos vs Target
        ax = axes[1, 0]

        loan_combinations = self.df.groupby(['housing', 'loan'])[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).reset_index()
        loan_combinations['combination'] = (
            loan_combinations['housing'].str.upper() + ' Housing\n' +
            loan_combinations['loan'].str.upper() + ' Loan'
        )

        bars = ax.bar(loan_combinations['combination'], loan_combinations[self.target_col],
                      color=self.colors['secondary'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax.set_title('Conversão por Tipo de Empréstimo',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=9)

        # 4. Inadimplência
        ax = axes[1, 1]

        default_counts = self.df.groupby(
            ['default', self.target_col]).size().unstack()
        default_pct = default_counts.div(
            default_counts.sum(axis=1), axis=0) * 100

        default_pct.plot(kind='bar',
                         ax=ax,
                         color=[self.colors['no'],
                                self.colors['yes']],
                         alpha=0.8,
                         edgecolor='black')
        ax.set_ylabel('Percentual (%)', fontsize=11)
        ax.set_xlabel('Inadimplente?', fontsize=11)
        ax.set_title('Target por Status de Inadimplência',
                     fontsize=12, fontweight='bold')
        ax.legend(title='Target', labels=['NO', 'YES'])
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()
        if save:
            plt.savefig('perfil_financeiro.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Perfil financeiro plotado")

    # INSIGHTS DE CAMPANHA

    def plot_analise_campanha(self, save: bool = False):
        """
        Análise detalhada da campanha de marketing: contatos, duração, timing.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Duração da ligação vs Conversão (MUITO IMPORTANTE!)
        ax1 = fig.add_subplot(gs[0, :2])

        duration_bins = [0, 100, 200, 300, 500, 1000, np.inf]
        duration_labels = ['0-100s', '101-200s',
                           '201-300s', '301-500s', '501-1000s', '1000s+']
        self.df['duration_cat'] = pd.cut(
            self.df['duration'], bins=duration_bins, labels=duration_labels)

        conversion_by_duration = self.df.groupby('duration_cat')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax1.bar(conversion_by_duration.index, conversion_by_duration.values,
                       color=self.colors['primary'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Taxa de Conversão (%)', fontsize=11)
        ax1.set_title('Duração da Ligação vs Conversão - incialmente considerada, se provou leakage',
                      fontsize=13, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.,
                     height,
                     f'{height:.1f}%',
                     ha='center',
                     va='bottom',
                     fontweight='bold',
                     fontsize=10)

        # 2. Número de Contatos vs Conversão
        ax2 = fig.add_subplot(gs[0, 2])

        campaign_bins = [0, 1, 2, 3, 5, 10, np.inf]
        campaign_labels = ['1', '2', '3', '4-5', '6-10', '10+']
        self.df['campaign_cat'] = pd.cut(
            self.df['campaign'], bins=campaign_bins, labels=campaign_labels)

        conversion_by_campaign = self.df.groupby('campaign_cat')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax2.bar(conversion_by_campaign.index,
                       conversion_by_campaign.values,
                       color=self.colors['warning'],
                       alpha=0.8,
                       edgecolor='black')
        ax2.set_ylabel('Taxa de Conversão (%)', fontsize=10)
        ax2.set_xlabel('Nº Contatos', fontsize=10)
        ax2.set_title('Contatos na Campanha', fontsize=11, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2.,
                         height,
                         f'{height:.1f}%',
                         ha='center',
                         va='bottom',
                         fontweight='bold',
                         fontsize=8)

        # 3. Sazonalidade: Mês vs Conversão
        ax3 = fig.add_subplot(gs[1, :])

        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        # Conversão por mês
        conversion_by_month = self.df.groupby('month')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).reindex(month_order)

        # Volume por mês (eixo secundário)
        volume_by_month = self.df['month'].value_counts().reindex(month_order)

        ax3_twin = ax3.twinx()

        # Linha de conversão
        line = ax3.plot(conversion_by_month.index, conversion_by_month.values,
                        color=self.colors['success'], linewidth=3, marker='o', markersize=8,
                        label='Taxa de Conversão')

        # Barras de volume
        bars = ax3_twin.bar(volume_by_month.index, volume_by_month.values,
                            alpha=0.3, color=self.colors['primary'], label='Volume de Contatos')

        ax3.set_ylabel('Taxa de Conversão (%)', fontsize=11,
                       color=self.colors['success'])
        ax3_twin.set_ylabel('Volume de Contatos', fontsize=11,
                            color=self.colors['primary'])
        ax3.set_xlabel('Mês', fontsize=11)
        ax3.set_title('Sazonalidade: Conversão e Volume por Mês',
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='y', labelcolor=self.colors['success'])
        ax3_twin.tick_params(axis='y', labelcolor=self.colors['primary'])

        # Adicionar valores na linha
        for x, y in zip(conversion_by_month.index, conversion_by_month.values):
            if not np.isnan(y):
                ax3.text(x, y + 0.5, f'{y:.1f}%', ha='center', va='bottom',
                         fontweight='bold', fontsize=8)

        # 4. Dia do Mês vs Conversão
        ax4 = fig.add_subplot(gs[2, 0])

        day_bins = [0, 10, 20, 31]
        day_labels = ['1-10', '11-20', '21-31']
        self.df['day_period'] = pd.cut(
            self.df['day'], bins=day_bins, labels=day_labels)

        conversion_by_day = self.df.groupby('day_period')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax4.bar(conversion_by_day.index, conversion_by_day.values,
                       color=self.colors['secondary'], alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Taxa de Conversão (%)', fontsize=10)
        ax4.set_xlabel('Período do Mês', fontsize=10)
        ax4.set_title('Conversão por Período', fontsize=11, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 5. Contatos Prévios vs Conversão
        ax5 = fig.add_subplot(gs[2, 1])

        self.df['previous_cat'] = pd.cut(self.df['previous'],
                                         bins=[-1, 0, 1, 3, np.inf],
                                         labels=['0', '1', '2-3', '4+'])

        conversion_by_previous = self.df.groupby('previous_cat')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax5.bar(conversion_by_previous.index, conversion_by_previous.values,
                       color=self.colors['success'], alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Taxa de Conversão (%)', fontsize=10)
        ax5.set_xlabel('Contatos Anteriores', fontsize=10)
        ax5.set_title('Histórico de Contatos', fontsize=11, fontweight='bold')
        ax5.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 6. Resultado de Campanha Anterior vs Conversão
        ax6 = fig.add_subplot(gs[2, 2])

        conversion_by_poutcome = self.df.groupby('poutcome')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False)

        bars = ax6.bar(conversion_by_poutcome.index, conversion_by_poutcome.values,
                       color=self.colors['danger'], alpha=0.8, edgecolor='black')
        ax6.set_ylabel('Taxa de Conversão (%)', fontsize=10)
        ax6.set_xlabel('Resultado Anterior', fontsize=10)
        ax6.set_title('Campanha Prévia', fontsize=11, fontweight='bold')
        ax6.grid(True, axis='y', alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

        if save:
            plt.savefig('analise_campanha.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Análise de campanha plotada")

    def plot_contact_type_analysis(self, save: bool = False):
        """
        Análise específica do tipo de contato.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Taxa de Conversão por Tipo de Contato
        ax = axes[0]

        conversion_by_contact = self.df.groupby('contact')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False)

        bars = ax.bar(conversion_by_contact.index, conversion_by_contact.values,
                      color=[self.colors['success'],
                             self.colors['warning'], self.colors['danger']],
                      alpha=0.8, edgecolor='black')
        ax.set_ylabel('Taxa de Conversão (%)', fontsize=12)
        ax.set_title('Conversão por Tipo de Contato',
                     fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 2. Volume vs Conversão por Tipo de Contato
        ax = axes[1]

        contact_stats = self.df.groupby('contact').agg({
            self.target_col: lambda x: (x == 'yes').mean() * 100,
            'age': 'count'  # usar qualquer coluna para contar
        }).rename(columns={'age': 'volume', self.target_col: 'conversion'})

        ax_twin = ax.twinx()

        # Barras de volume
        bars = ax.bar(contact_stats.index, contact_stats['volume'],
                      alpha=0.6, color=self.colors['primary'], label='Volume')

        # Linha de conversão
        line = ax_twin.plot(contact_stats.index, contact_stats['conversion'],
                            color=self.colors['danger'], linewidth=3, marker='o',
                            markersize=10, label='Conversão')

        ax.set_ylabel('Volume de Contatos', fontsize=11,
                      color=self.colors['primary'])
        ax_twin.set_ylabel('Taxa de Conversão (%)',
                           fontsize=11, color=self.colors['danger'])
        ax.set_title('Volume vs Efetividade por Canal',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax_twin.tick_params(axis='y', labelcolor=self.colors['danger'])

        # Adicionar valores
        for i, (idx, row) in enumerate(contact_stats.iterrows()):
            ax_twin.text(i, row['conversion'] + 0.5, f"{row['conversion']:.1f}%",
                         ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        if save:
            plt.savefig('contact_type_analysis.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Análise de tipo de contato plotada")

    # DASHBOARD EXECUTIVO

    def create_executive_dashboard(self, save: bool = False):
        """
        Cria um dashboard executivo com os principais KPIs e insights.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        # Calcular KPIs principais
        total_contacts = len(self.df)
        conversions = (self.df[self.target_col] == 'yes').sum()
        conversion_rate = (conversions / total_contacts) * 100

        avg_duration = self.df['duration'].mean()
        avg_campaign = self.df['campaign'].mean()
        avg_balance = self.df['balance'].mean()

        # === SEÇÃO 1: KPIs PRINCIPAIS ===
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.7, 'BANK MARKETING CAMPAIGN',
                      ha='center', va='center', fontsize=24, fontweight='bold')
        ax_title.text(0.5, 0.3, 'Executive Dashboard - Key Performance Indicators',
                      ha='center', va='center', fontsize=14, style='italic')

        # KPI Cards
        kpi_data = [
            ('Total Contatos', f'{total_contacts:,}', self.colors['primary']),
            ('Conversões', f'{conversions:,}', self.colors['success']),
            ('Taxa Conversão', f'{conversion_rate:.2f}%',
             self.colors['warning']),
            ('Duração Média', f'{avg_duration:.0f}s', self.colors['secondary'])
        ]

        for idx, (label, value, color) in enumerate(kpi_data):
            ax_kpi = fig.add_subplot(gs[1, idx])
            ax_kpi.axis('off')

            # Box colorido
            box = plt.Rectangle((0.1, 0.2), 0.8, 0.6,
                                facecolor=color, alpha=0.3, edgecolor=color, linewidth=3)
            ax_kpi.add_patch(box)

            ax_kpi.text(0.5, 0.65, value, ha='center', va='center',
                        fontsize=22, fontweight='bold', color=color)
            ax_kpi.text(0.5, 0.35, label, ha='center', va='center',
                        fontsize=11, color='black')

        # SEÇÃO 2: TOP INSIGHTS

        # 1. Top 5 Profissões
        ax1 = fig.add_subplot(gs[2, 0])
        top_jobs = self.df.groupby('job')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False).head(5)

        bars = ax1.barh(top_jobs.index, top_jobs.values,
                        color=self.colors['primary'], alpha=0.7)
        ax1.set_xlabel('Conv. (%)', fontsize=9)
        ax1.set_title('Top 5 Profissões', fontsize=11, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                     f'{width:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')

        # 2. Melhor Faixa Etária
        ax2 = fig.add_subplot(gs[2, 1])
        age_bins = [18, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        self.df['age_seg'] = pd.cut(
            self.df['age'], bins=age_bins, labels=age_labels)

        conversion_by_age = self.df.groupby('age_seg')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax2.bar(conversion_by_age.index, conversion_by_age.values,
                       color=self.colors['success'], alpha=0.7)
        ax2.set_ylabel('Conv. (%)', fontsize=9)
        ax2.set_title('Conversão por Idade', fontsize=11, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(),
                 rotation=45, ha='right', fontsize=8)

        # 3. Saldo vs Conversão
        ax3 = fig.add_subplot(gs[2, 2])
        balance_bins = [-np.inf, 0, 1000, 5000, np.inf]
        balance_labels = ['Neg.', '0-1k', '1k-5k', '5k+']
        self.df['balance_seg'] = pd.cut(
            self.df['balance'], bins=balance_bins, labels=balance_labels)

        conversion_by_balance = self.df.groupby('balance_seg')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        )

        bars = ax3.bar(conversion_by_balance.index, conversion_by_balance.values,
                       color=self.colors['warning'], alpha=0.7)
        ax3.set_ylabel('Conv. (%)', fontsize=9)
        ax3.set_title('Conversão por Saldo', fontsize=11, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)

        # 4. Melhor Canal
        ax4 = fig.add_subplot(gs[2, 3])
        conversion_by_contact = self.df.groupby('contact')[self.target_col].apply(
            lambda x: (x == 'yes').mean() * 100
        ).sort_values(ascending=False)

        bars = ax4.bar(conversion_by_contact.index, conversion_by_contact.values,
                       color=self.colors['danger'], alpha=0.7)
        ax4.set_ylabel('Conv. (%)', fontsize=9)
        ax4.set_title('Melhor Canal', fontsize=11, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        if save:
            plt.savefig('executive_dashboard.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Dashboard executivo criado")

    #
    # PIPELINE COMPLETO

    # para facilitar a minha vida no controle do alpha
    def with_alpha(color, alpha):
        return to_rgba(color, alpha)

    def gerar_dashboard_resumo(self, cat_col: str = "job"):
        """
        Gera um resumo executivo da campanha:
        - Taxa de conversão global
        - Top 3 categorias com maior conversão em uma variável categórica
        - Correlações mais fortes com o target
        - Gráfico de conversão por categoria

        Args:
            cat_col: coluna categórica de referência para análise (ex: 'job')
        """
        df = self.df.copy()
        target = self.target_col

        # Taxa global
        taxa_global = (df[target] == 'yes').mean() * 100
        print(f"Taxa global de conversão: {taxa_global:.2f}%")

        # Conversão por categoria
        conv_categoria = df.groupby(cat_col)[target].apply(
            lambda x: (x == 'yes').mean()).sort_values(ascending=False) * 100

        print(f"\nTop 3 categorias em '{cat_col}' com maior conversão:")
        print(conv_categoria.head(3).round(2).astype(str) + "%")

        # Correlações com target (numéricas)
        df[target + "_bin"] = (df[target] == "yes").astype(int)
        corr = df[self.numeric_cols + [target + "_bin"]].corr()[target +
                                                                "_bin"].drop(target + "_bin").sort_values(key=abs, ascending=False)

        print("\nVariáveis numéricas mais correlacionadas com o target:")
        print(corr.head(5).round(3))

        # --- Gráfico ---
        plt.figure(figsize=(10, 6))
        sns.barplot(x=conv_categoria.index,
                    y=conv_categoria.values, palette="Blues_r")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Taxa de Conversão (%)")
        plt.xlabel(cat_col.capitalize())
        plt.title(
            f"Taxa de Conversão por {cat_col.capitalize()} (Global: {taxa_global:.1f}%)", fontsize=14, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def gerar_todas_visualizacoes(self, save_all: bool = True):
        """
        Gera todas as visualizações de uma vez.

        Args:
            save_all: salvar todas as figuras
        """

        print("\n" + "" * 30)
        print("GERANDO VISUALIZAÇÕES COMPLETAS")
        print("" * 30 + "\n")

        print("\n1. ANÁLISE UNIVARIADA")
        print("-" * 60)
        self.plot_distribuicao_numericas(save=save_all)
        self.plot_distribuicao_categoricas(save=save_all)

        print("\n2. ANÁLISE BIVARIADA (vs TARGET)")
        print("-" * 60)
        self.plot_numericas_vs_target(save=save_all)
        self.plot_categoricas_vs_target(save=save_all)

        print("\n3. CORRELAÇÕES")
        print("-" * 60)
        self.plot_correlation_matrix(save=save_all)
        self.plot_vif_analysis(save=save_all)

        print("\n4. SEGMENTAÇÃO")
        print("-" * 60)
        self.plot_segmentacao_demografica(save=save_all)
        self.plot_perfil_financeiro(save=save_all)

        print("\n5. ANÁLISE DE CAMPANHA")
        print("-" * 60)
        self.plot_analise_campanha(save=save_all)
        self.plot_contact_type_analysis(save=save_all)

        print("\n6. DASHBOARD EXECUTIVO")
        print("-" * 60)
        self.create_executive_dashboard(save=save_all)

        self.gerar_dashboard_resumo(cat_col="job")

        print("\n" + "" * 30)
        print("TODAS AS VISUALIZAÇÕES GERADAS COM SUCESSO!")
        print("" * 30 + "\n")
