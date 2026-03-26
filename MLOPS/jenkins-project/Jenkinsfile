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
                // Создаём виртуальное окружение и ставим зависимости
                sh 'python3 -m venv venv'
                sh 'venv/bin/pip install --upgrade pip setuptools'
                sh 'venv/bin/pip install -r requirements.txt'
            }
        }

        stage('Preprocess') {
            steps {
                // Download dataset from Kaggle
                withCredentials([string(credentialsId: 'kaggle-json', variable: 'KAGGLE_JSON')]) {
                    sh '''#!/bin/bash
                    mkdir -p ~/.kaggle
                    echo "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
                    chmod 600 ~/.kaggle/kaggle.json
                    
                    venv/bin/pip install --no-cache-dir kaggle
                    venv/bin/python scripts/download_kaggle.py
                    '''
                }
                // Локальный препроцессинг
                sh 'PYTHONPATH=. venv/bin/python scripts/preprocess.py'
            }
        }

        stage('Train') {
            steps {
                sh 'PYTHONPATH=. venv/bin/python scripts/train.py'
            }
            post {
                success {
                    archiveArtifacts artifacts: 'model_bundle.pkl, lr_pipeline.pkl, best_model.txt', fingerprint: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                // Собираем образ с релевантным именем (amazon-sales)
                sh "docker build -t ml-amazon-sales:${env.BUILD_NUMBER} ."
            }
        }

        stage('Run Service') {
            steps {
                sh "docker rm -f ml-amazon-sales || true"
                // Пробрасываем на порт 8585, чтобы обойти любые старые зависшие процессы Flask/MLflow
                sh "docker run -d --name ml-amazon-sales -p 8585:80 ml-amazon-sales:${env.BUILD_NUMBER}"
            }
        }

        stage('Smoke Test') {
            steps {
                // Ждем инициализации
                sh 'sleep 5'
                
                sh '''#!/bin/bash
                # Обращаемся на наш новый чистый порт 8585
                RESPONSE=$(curl -sS -X POST http://127.0.0.1:8585/predict \
                -H "Content-Type: application/json" \
                -d '{"inputs":[{"price": 100.0, "discount_percent": 5.0, "quantity_sold": 2, "rating": 4.5, "review_count": 100, "product_category": 1, "customer_region": 1, "payment_method": 1, "order_year": 2023, "order_month": 5}]}')

                echo "Response from API: $RESPONSE"
                
                if [[ "$RESPONSE" == *"error"* ]] || [[ "$RESPONSE" == *"405 Not Allowed"* ]] || [[ "$RESPONSE" == *"Not Found"* ]] || [[ -z "$RESPONSE" ]]; then
                    echo "Smoke test failed! API returned an error."
                    docker logs ml-amazon-sales
                    exit 1
                fi
                
                if [[ "$RESPONSE" == *"predictions"* ]]; then
                    echo "Smoke test PASSED! Model is working."
                else
                    echo "Unexpected response format!"
                    docker logs ml-amazon-sales
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
