-- schema.sql
-- Banco de dados para rastreamento de experimentos de classificação de imagens.
--
-- Uso:
--     sqlite3 experiments.db < schema.sql
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────
-- Tabela principal: uma linha por run
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    run_name TEXT NOT NULL,
    UNIQUE (experiment, run_name)
);

-- ─────────────────────────────────────────────
-- Parâmetros da run (config + decision threshold)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS run_params (
    run_id INTEGER PRIMARY KEY,
    -- Modelo
    base_model TEXT,
    weights TEXT,
    -- Pré-processamento
    normalization TEXT,
    data_aug BOOLEAN,
    -- Imagem
    image_size TEXT,
    -- Treinamento
    optimizer_name TEXT,
    learning_rate REAL,
    batch_size INTEGER,
    epochs INTEGER,
    seed INTEGER,
    class_weights BOOLEAN,
    --
    -- Avaliação
    decision_threshold REAL,
    FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE
);

-- ─────────────────────────────────────────────
-- Métricas de teste da run
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id INTEGER PRIMARY KEY,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    specificity REAL,
    auc_roc REAL,
    FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE
);

-- ─────────────────────────────────────────────
-- Views utilitárias
-- ─────────────────────────────────────────────
-- Join completo: uma linha por run com params + metrics
CREATE VIEW IF NOT EXISTS v_runs_full AS
SELECT
    r.id,
    r.experiment,
    r.run_name,
    p.base_model,
    p.weights,
    p.normalization,
    p.data_aug,
    p.image_size,
    p.optimizer_name,
    p.learning_rate,
    p.batch_size,
    p.epochs,
    p.seed,
    p.class_weights,
    p.decision_threshold,
    m.accuracy,
    m.precision,
    m.recall,
    m.f1_score,
    m.specificity,
    m.auc_roc
FROM
    runs r
    LEFT JOIN run_params p ON p.run_id = r.id
    LEFT JOIN run_metrics m ON m.run_id = r.id;

-- Média e desvio padrão por experimento
-- Nota: SQLite não possui STDDEV nativo.
-- Usa a identidade: std = sqrt(avg(x²) - avg(x)²)
CREATE VIEW IF NOT EXISTS v_experiment_summary AS
SELECT
    experiment,
 
    COUNT(*) AS n_runs,
 
    ROUND(AVG(accuracy), 4) AS accuracy_mean,
    ROUND(SQRT(AVG(accuracy * accuracy) - AVG(accuracy) * AVG(accuracy)), 4) AS accuracy_std,
 
    ROUND(AVG(precision), 4) AS precision_mean,
    ROUND(SQRT(AVG(precision * precision) - AVG(precision) * AVG(precision)), 4) AS precision_std,
 
    ROUND(AVG(recall), 4) AS recall_mean,
    ROUND(SQRT(AVG(recall * recall) - AVG(recall) * AVG(recall)), 4) AS recall_std,
 
    ROUND(AVG(f1_score), 4) AS f1_mean,
    ROUND(SQRT(AVG(f1_score * f1_score) - AVG(f1_score) * AVG(f1_score)), 4) AS f1_std,
 
    ROUND(AVG(specificity), 4) AS specificity_mean,
    ROUND(SQRT(AVG(specificity * specificity) - AVG(specificity) * AVG(specificity)), 4) AS specificity_std,
 
    ROUND(AVG(auc_roc), 4) AS auc_roc_mean,
    ROUND(SQRT(AVG(auc_roc * auc_roc) - AVG(auc_roc) * AVG(auc_roc)), 4) AS auc_roc_std
 
FROM v_runs_full
GROUP BY experiment;