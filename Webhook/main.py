import os
import json
import subprocess
import google.generativeai as genai
from flask import Flask, request, jsonify
import requests

# Create Flask app
app = Flask(__name__)

# Set your API keys here
API_KEY = ""
# Replace with a Bot User OAuth Token (starts with xoxb-)
SLACK_TOKEN = ""
SLACK_CHANNEL = ""  

def send_to_slack(message, blocks=None):
    """Send a message to Slack channel"""
    print("\n=== Sending to Slack ===")
    
    headers = {
        "Authorization": f"Bearer {SLACK_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "channel": SLACK_CHANNEL,
        "text": message
    }
    
    if blocks:
        payload["blocks"] = blocks
    
    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload
        )
        result = response.json()
        
        if result.get("ok"):
            print("Message sent to Slack successfully!")
        else:
            print(f"Failed to send message to Slack: {result.get('error')}")
        
        return result
    except Exception as e:
        print(f"Exception sending to Slack: {e}")
        return None

def format_slack_alert(alert_info, analysis=None):
    """Format alert information for Slack with rich formatting"""
    blocks = []
    
    # Alert Header
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"🚨 Kubernetes Alert: {alert_info.get('alertname', 'Unknown Issue')}"
        }
    })
    
    alert_fields = []
    if alert_info.get('namespace'):
        alert_fields.append({
            "type": "mrkdwn",
            "text": f"*Namespace:*\n{alert_info.get('namespace')}"
        })
    if alert_info.get('pod'):
        alert_fields.append({
            "type": "mrkdwn",
            "text": f"*Pod:*\n{alert_info.get('pod')}"
        })
    if alert_info.get('container'):
        alert_fields.append({
            "type": "mrkdwn",
            "text": f"*Container:*\n{alert_info.get('container')}"
        })
    if alert_info.get('reason'):
        alert_fields.append({
            "type": "mrkdwn",
            "text": f"*Reason:*\n{alert_info.get('reason')}"
        })
    
    # Only add section if we have fields
    if alert_fields:
        blocks.append({
            "type": "section",
            "fields": alert_fields[:10]
        })
    
    if alert_info.get('description') or alert_info.get('summary'):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Description:*\n{alert_info.get('description') or alert_info.get('summary', 'No description available')}"
            }
        })
    
    blocks.append({"type": "divider"})
    
    if analysis:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Issue:*\n{analysis.get('issue', 'Unknown issue')}"
            }
        })
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Explanation:*\n{analysis.get('explanation', 'No explanation available')}"
            }
        })
        
        if analysis.get('solutions'):
            for i, solution in enumerate(analysis.get('solutions')):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Solution {i+1}: {solution.get('title')}*\n{solution.get('description')}"
                    }
                })
                if solution.get('commands'):
                    command_text = "```\n" + "\n".join(solution.get('commands')) + "\n```"
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Commands:*\n{command_text}"
                        }
                    })
                if solution.get('yaml'):
                    yaml_text = "```yaml\n" + solution.get('yaml') + "\n```"
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*YAML:*\n{yaml_text}"
                        }
                    })
    
    return blocks

def run_k8sgpt_analyze(namespace=None, pod=None):
    """Run k8sgpt analyze with optional filters for namespace and pod"""
    print(f"=== Running k8sgpt analyze for namespace={namespace}, pod={pod} ===")
    
    try:
        cmd = ["k8sgpt", "analyze"]
        if namespace:
            cmd.extend(["--namespace", namespace])
        if pod:
            cmd.extend(["--filter", f"Pod={pod}"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running k8sgpt: {result.stderr}")
            return None
        
        output = result.stdout.strip()
        print("\n=== K8sGPT Analysis Output ===")
        print(output)
        return output
    
    except Exception as e:
        print(f"Exception running k8sgpt: {e}")
        return None

def ask_gemini(k8sgpt_output, alert_data=None):
    """Query Gemini AI with k8sgpt output and optional alert data"""
    print("\n=== Consulting Gemini AI ===")
    if not API_KEY:
        print("Error: Gemini API_KEY is not set.")
        return None
    
    # Configure Gemini API key
    genai.configure(api_key=API_KEY)
    
    # Create model with specific configuration
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Request JSON response
    }
    
    # Use available model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config
    )
    
    # Include alert information in the prompt if available
    alert_info = ""
    if alert_data:
        alert_info = f"""
        
Alert information from Prometheus:
Alert Name: {alert_data.get('alertname', 'Unknown')}
Namespace: {alert_data.get('namespace', 'Unknown')}
Pod: {alert_data.get('pod', 'Unknown')}
Container: {alert_data.get('container', 'Unknown')}
Reason: {alert_data.get('reason', 'Unknown')}
Description: {alert_data.get('description', 'No description available')}
Summary: {alert_data.get('summary', 'No summary available')}
"""
    
    # Formulating the prompt for Gemini with JSON structure guidance
    prompt = f"""
I'm working with Kubernetes and encountered the following issue detected by k8sgpt:
{k8sgpt_output}{alert_info}

Please provide your analysis in the following JSON format exactly:
{{
  "issue": "Brief description of the issue",
  "explanation": "Simple explanation of what's causing the issue",
  "solutions": [
    {{
      "title": "Solution 1 title",
      "description": "Detailed description of solution 1",
      "commands": [
        "command 1",
        "command 2"
      ]
    }},
    {{
      "title": "Solution 2 title",
      "description": "Detailed description of solution 2",
      "yaml": "example YAML if applicable"
    }}
  ]
}}

The response should be valid JSON, properly escaped, with no additional text outside the JSON structure.
"""
    
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        else:
            print("\nNo valid response from Gemini.")
            return None
    except Exception as e:
        print(f"Error connecting to Gemini AI: {e}")
        return None

def save_analysis_result(alert_data, k8sgpt_output, gemini_response):
    """Save analysis results to a file for later reference"""
    try:
        # Create a results directory if it doesn't exist
        os.makedirs("analysis_results", exist_ok=True)
        
        # Generate a filename based on alert info
        namespace = alert_data.get('namespace', 'unknown')
        pod = alert_data.get('pod', 'unknown')
        timestamp = alert_data.get('startsAt', '').replace(':', '-').replace('.', '-') if alert_data.get('startsAt') else ''
        
        filename = f"analysis_results/alert_{namespace}_{pod}_{timestamp}.json"
        
        # Prepare the data to save
        result = {
            "alert_data": alert_data,
            "k8sgpt_output": k8sgpt_output,
            "gemini_analysis": json.loads(gemini_response) if gemini_response else None,
            "timestamp": alert_data.get('startsAt')
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Analysis results saved to {filename}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")

# Support both the original webhook endpoint and the simplified alert endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook requests from AlertManager"""
    try:
        # Get the JSON data from the request
        alert_data = request.json
        print("\n=== Received Alert via /webhook ===")
        print(json.dumps(alert_data, indent=2))
        
        # Check if this is a valid alert
        if not alert_data or 'alerts' not in alert_data or not alert_data['alerts']:
            return jsonify({"status": "error", "message": "No valid alerts in payload"}), 400
        
        # Process the first alert (can be modified to handle multiple alerts)
        alert = alert_data['alerts'][0]
        labels = alert.get('labels', {})
        annotations = alert.get('annotations', {})
        
        # Extract relevant information
        alert_info = {
            "alertname": labels.get('alertname'),
            "namespace": labels.get('namespace'),
            "pod": labels.get('pod'),
            "container": labels.get('container'),
            "reason": labels.get('reason'),
            "description": annotations.get('description'),
            "summary": annotations.get('summary'),
            "startsAt": alert.get('startsAt')
        }
        
        # Run k8sgpt analysis with filters based on the alert
        k8sgpt_output = run_k8sgpt_analyze(
            namespace=labels.get('namespace'),
            pod=labels.get('pod')
        )
        
        response_data = {"status": "processing", "alert_info": alert_info}
        
        if k8sgpt_output:
            # Get analysis from Gemini
            gemini_response = ask_gemini(k8sgpt_output, alert_info)
            
            if gemini_response:
                try:
                    # Parse and add to response
                    analysis = json.loads(gemini_response)
                    response_data["analysis"] = analysis
                    response_data["status"] = "success"
                    
                    # Save results
                    save_analysis_result(alert_info, k8sgpt_output, gemini_response)
                    
                    # Send to Slack
                    blocks = format_slack_alert(alert_info, analysis)
                    slack_message = f"Kubernetes Alert: {alert_info.get('alertname', 'Unknown Issue')} in {alert_info.get('namespace', 'unknown')}/{alert_info.get('pod', 'unknown')}"
                    send_to_slack(slack_message, blocks)
                    
                except json.JSONDecodeError:
                    response_data["status"] = "error"
                    response_data["message"] = "Failed to parse Gemini response"
                    
                    # Send error message to Slack
                    send_to_slack(f"⚠️ Error analyzing Kubernetes alert: Failed to parse Gemini response for {alert_info.get('namespace', 'unknown')}/{alert_info.get('pod', 'unknown')}")
            else:
                response_data["status"] = "error"
                response_data["message"] = "No response from Gemini"
                send_to_slack(f"⚠️ Error analyzing Kubernetes alert: No response from Gemini for {alert_info.get('namespace', 'unknown')}/{alert_info.get('pod', 'unknown')}")
        else:
            response_data["status"] = "error"
            response_data["message"] = "k8sgpt analysis failed"
            send_to_slack(f"⚠️ Error analyzing Kubernetes alert: k8sgpt analysis failed for {alert_info.get('namespace', 'unknown')}/{alert_info.get('pod', 'unknown')}")
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing webhook: {e}")
        send_to_slack(f"❌ Critical error in Kubernetes alert processing: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/alert', methods=['POST'])
def handle_alert():
    """Handle simplified alert endpoint"""
    try:
        data = request.get_json()
        print("\n=== Received Alert via /alert ===")
        print(json.dumps(data, indent=2))
        
        if not data:
            return jsonify({"status": "error", "message": "No data in request"}), 400
        
        k8sgpt_output = run_k8sgpt_analyze(data.get("namespace"), data.get("pod"))
        
        if not k8sgpt_output:
            message = f"⚠️ Error analyzing Kubernetes alert: k8sgpt analysis failed for {data.get('namespace', 'unknown')}/{data.get('pod', 'unknown')}"
            send_to_slack(message)
            return jsonify({"status": "error", "message": "k8sgpt analysis failed"}), 500
        
        gemini_response = ask_gemini(k8sgpt_output, data)
        
        if not gemini_response:
            message = f"⚠️ Error analyzing Kubernetes alert: No response from Gemini for {data.get('namespace', 'unknown')}/{data.get('pod', 'unknown')}"
            send_to_slack(message)
            return jsonify({"status": "error", "message": "No response from Gemini"}), 500
        
        try:
            # Attempt to parse the Gemini response as JSON
            analysis = json.loads(gemini_response)
            
            # Format and send to Slack
            blocks = format_slack_alert(data, analysis)
            message = f"Kubernetes Alert: {data.get('alertname', 'Unknown Issue')} in {data.get('namespace', 'unknown')}/{data.get('pod', 'unknown')}"
            send_to_slack(message, blocks)
            
            # Save the analysis results
            save_analysis_result(data, k8sgpt_output, gemini_response)
            
            return jsonify({
                "status": "success", 
                "alert_info": data,
                "analysis": analysis
            })
            
        except json.JSONDecodeError as e:
            message = f"⚠️ Error analyzing Kubernetes alert: Invalid JSON response from Gemini for {data.get('namespace', 'unknown')}/{data.get('pod', 'unknown')}"
            send_to_slack(message)
            return jsonify({"status": "error", "message": f"Invalid JSON from Gemini: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error handling alert: {e}")
        send_to_slack(f"❌ Critical error in Kubernetes alert processing: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to verify service is running"""
    return jsonify({
        "status": "running",
        "service": "K8s Alert Service",
        "endpoints": {
            "/webhook": "AlertManager webhook endpoint",
            "/alert": "Simplified alert endpoint",
            "/health": "Health check endpoint"
        }
    })

def test_slack_connection():
    """Test the Slack connection at startup"""
    if SLACK_TOKEN and SLACK_TOKEN != "":
        test_result = send_to_slack("🚀 K8s Alert System is now online and monitoring your cluster")
        if test_result and test_result.get("ok"):
            print("✅ Successfully connected to Slack")
        else:
            print(f"❌ Failed to connect to Slack: {test_result.get('error') if test_result else 'Unknown error'}")
    else:
        print("⚠️ Slack integration is disabled (no token configured)")

if __name__ == "__main__":
    # Check if k8sgpt is installed
    try:
        k8sgpt_path = subprocess.run(
            ["which", "k8sgpt"], 
            capture_output=True, 
            text=True
        ).stdout.strip()
        
        if k8sgpt_path:
            print(f"Found k8sgpt at: {k8sgpt_path}")
        else:
            print("WARNING: k8sgpt not found in PATH. Please install it before running this service.")
    except Exception as e:
        print(f"Error locating k8sgpt: {e}")
    
    # Test Slack connection
    test_slack_connection()
    
    # Start the Flask server
    print("Starting webhook server on port 5000...")
    app.run(host='0.0.0.0', port=5000)