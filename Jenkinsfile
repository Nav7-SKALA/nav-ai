pipeline {
    agent any

    environment {
        GIT_URL = 'https://github.com/Nav7-SKALA/nav-ai.git'
        GIT_BRANCH = 'main'
        GIT_ID = 'skala-github-id'
        IMAGE_REGISTRY = 'amdp-registry.skala-ai.com/skala25a'
        IMAGE_NAME = 'sk-nav7-ai'
        IMAGE_TAG = '1.0.0'
        DOCKER_CREDENTIAL_ID = 'skala-image-registry-id'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: "${GIT_BRANCH}",
                    url: "${GIT_URL}",
                    credentialsId: "${GIT_ID}"
            }
        }

        stage('Docker Build & Push') {
            steps {
                script {
                    def FINAL_IMAGE_TAG = "${IMAGE_TAG}-${BUILD_NUMBER}"
                    
                    docker.withRegistry("https://${IMAGE_REGISTRY}", "${DOCKER_CREDENTIAL_ID}") {
                        def appImage = docker.build("${IMAGE_REGISTRY}/${IMAGE_NAME}:${FINAL_IMAGE_TAG}", "./api")
                        appImage.push()
                    }
                    
                    env.FINAL_IMAGE_TAG = FINAL_IMAGE_TAG
                }
            }
        }

        stage('Update Deploy Config') {
            steps {
                sh """
                    sed -i 's|image:.*|image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${env.FINAL_IMAGE_TAG}|g' ./k8s/deploy.yaml
                """
            }
        }
    }
}