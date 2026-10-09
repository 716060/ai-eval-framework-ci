pipeline {
    agent none

    // ========== 全局环境变量 ==========
    environment {
        DEEPEVAL_TELEMETRY_OPT_OUT = "YES"
    }

    // ========== 触发条件 ==========
    triggers {
        // 每天凌晨 2:00（北京时间）自动运行一次
        cron('0 2 * * *')
    }

    options {
        // 保留最近 10 次构建记录
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // 构建超时时间（大模型评测可能很慢，设为 1 小时）
        timeout(time: 1, unit: 'HOURS')
        // 显示时间戳
        timestamps()
        // 禁止并发构建（避免 API Key 被同时占用）
        disableConcurrentBuilds()
    }

    stages {

        // ========== Stage 1: 代码拉取 ==========
        stage('Checkout') {
            agent any  // 用 Jenkins 主容器拉代码即可
            steps {
                echo "📥 拉取代码..."
                checkout scm
                sh 'git log -1 --oneline'
            }
        }

        // ========== Stage 2: 静态检查 + 单元测试 ==========
        stage('Lint & Test') {
            agent {
                docker {
                    image 'python:3.10-slim'
                    args '-u root:root --network host'
                    reuseNode true
                }
            }
            steps {
                echo "🔍 执行静态检查与单元测试..."
                sh '''
                    set -e
                    pip install --upgrade pip
                    pip install -e ".[all]"
                    ruff check . && ruff format --check .
                    mypy core config utils main.py
                    mkdir -p reports
                    pytest -q --junitxml=reports/pytest.xml -n auto
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/pytest.xml'
                }
            }
        }

        // ========== Stage 3: 冒烟评测（快速反馈）==========
        stage('Smoke Eval') {
            agent {
                docker {
                    image 'python:3.10-slim'
                    args '-u root:root --network host'
                    reuseNode true
                }
            }
            steps {
                echo "💨 执行冒烟评测..."
                withCredentials([
                    string(credentialsId: 'deepseek-api-key', variable: 'DEEPSEEK_API_KEY'),
                    string(credentialsId: 'qwen-api-key', variable: 'QWEN_API_KEY')
                ]) {
                    sh '''
                        set -e
                        pip install --upgrade pip
                        pip install -e ".[all]"
                        python main.py --smoke --tag P0 --limit 20 --checkpoint-dir .ckpt
                    '''
                }
            }
        }

        // ========== Stage 4: 全量评测 + 基线回归 ==========
        stage('Full Eval & Baseline') {
            agent {
                docker {
                    image 'python:3.10-slim'
                    args '-u root:root --network host'
                    reuseNode true
                }
            }
            steps {
                echo "🎯 执行全量评测与基线对比..."
                withCredentials([
                    string(credentialsId: 'deepseek-api-key', variable: 'DEEPSEEK_API_KEY'),
                    string(credentialsId: 'qwen-api-key', variable: 'QWEN_API_KEY')
                ]) {
                    sh '''
                        set -e
                        pip install --upgrade pip
                        pip install -e ".[all]"
                    '''
                    // catchError：让评测失败不立刻终止 Pipeline，而是标记 FAILURE 后继续
                    catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                        sh '''
                            python main.py --baseline reports/baseline.json --max-regression 0.02 --workers 8
                        '''
                    }
                }
            }
            post {
                always {
                    echo "📦 归档评测报告..."
                    archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true, fingerprint: true
                }
            }
        }
    }

    // ========== Pipeline 结束后的最终判定 ==========
    post {
        success {
            echo "✅ 门禁通过：所有评测指标符合基线要求"
        }
        failure {
            echo "❌ 门禁失败：评测未通过或发生回归，请查看 reports/ 目录"
        }
        aborted {
            echo "⚠️ 构建被中止"
        }
    }
}