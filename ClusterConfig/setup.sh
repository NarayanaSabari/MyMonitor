helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update

kubectl create namespace monitoring

helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring -f prometheus-values.yaml

kubectl apply -f pod-monitoring-rules.yaml  

kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n monitoring 9090:9090

kubectl port-forward svc/alertmanager-operated -n monitoring 9093:9093

