pipeline {
    agent {
        docker { image 'python:3.10-slim' } // 直接用一个带 Python 的镜像
    }
    stages {
        stage('Checkout') {
            steps {
                // 这里填你的 Git 仓库地址和分支
                git branch: 'main', url: 'https://github.com/716060/ai-eval-framework-ci.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                // 1. 创建虚拟环境（强烈建议，避免污染 Jenkins 主机的 Python 环境）
                sh 'python3 -m venv venv'

                // 2. 激活虚拟环境并安装依赖
                // 注意：这里使用 ".[dev]" 来安装 pyproject.toml 中定义的 dev 分组依赖
                // 如果你的项目没有打包成可安装的包，也可以直接用 "pip install -e '.[dev]'"
                sh '''
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install .[dev]
                '''
            }
        }

        stage('Run Tests') {
            steps {
                // 激活环境并运行 pytest
                // --junitxml 是为了让 Jenkins 能读取测试报告
                sh '''
                    . venv/bin/activate
                    pytest tests/ --junitxml=reports/results.xml
                '''
            }
        }
    }

    post {
        always {
            // 收集测试报告，这样 Jenkins 界面上才能看到测试结果和趋势图
            junit 'reports/results.xml'
        }
    }
}