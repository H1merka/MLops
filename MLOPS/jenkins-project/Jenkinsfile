pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup') {
            steps {
                // создаём виртуальное окружение и ставим зависимости
                sh 'python3 -m venv venv'
                sh '. venv/bin/activate && pip install --upgrade pip setuptools'
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Preprocess') {
            steps {
                // download dataset from Kaggle (kaggle.json must be stored in Jenkins credentials as 'kaggle-json')
                withCredentials([string(credentialsId: 'kaggle-json', variable: 'KAGGLE_JSON')]) {
                    sh '''
                    mkdir -p ~/.kaggle
                    echo "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
                    chmod 600 ~/.kaggle/kaggle.json
                    . venv/bin/activate
                    pip install --no-cache-dir kaggle
                    python3 scripts/download_kaggle.py
                    '''
                }
                // затем локальный препроцессинг
                sh 'PYTHONPATH=. venv/bin/activate && python3 scripts/preprocess.py'
            }
        }

        stage('Train') {
            steps {
                sh 'PYTHONPATH=. venv/bin/activate && python3 scripts/train.py'
            }
            post {
                success {
                    archiveArtifacts artifacts: 'model_bundle.pkl, lr_pipeline.pkl, best_model.txt', fingerprint: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                // Собираем образ для сервиса (если Jenkins имеет доступ к Docker)
                sh "docker build -t ml-cars:${env.BUILD_NUMBER} ."
            }
        }

        stage('Run Service') {
            steps {
                // Запускаем контейнер с моделью (рестартит старый, если был)
                sh "docker rm -f ml-cars || true || true"
                sh "docker run -d --name ml-cars -p 80:80 ml-cars:${env.BUILD_NUMBER}"
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                RESPONSE=$(curl -sS -X POST http://127.0.0.1/predict -H "Content-Type: application/json" \
                -d '{"inputs":[{"price": 100.0, "discount_percent": 5.0, "quantity_sold": 2, "rating": 4.5, "review_count": 100, "product_category": 1, "customer_region": 1, "payment_method": 1, "order_year": 2023, "order_month": 5}]}')

                echo "Response: $RESPONSE"
                if [[ "$RESPONSE" == *"error"* ]]; then
                    echo "Smoke test failed!"
                    exit 1
                fi
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'best_model.txt', allowEmptyArchive: true
        }
    }
}
