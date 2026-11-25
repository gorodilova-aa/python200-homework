# Part 3: Project

## Part A: Supabase Setup

* Step 1
    Created my Account and Project python-200
* Step 2
    Located my Credentials URL and anon key
* Step 3
    Added .env to assignments_09 repo
* Step 4
    Created the Tables weather_raw and weather_enriched
* Step 5
    Confirmed that two tables correctly created

## Part B: Cloud Cost Analysis

### Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

Scenario A (t3.micro): $1.66 / month ($19.97 / year).

### Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

Scenario B (p3.2xlarge): $489.60 / month ($5,875.20 / year).

#### Reflection 
The nearly 300x price difference for the exact same 160 hours of monthly runtime is surprising, clearly showing how fast compute costs jump once dedicated GPUs are involved.

The ~$488/month cost difference shows that GPU instances are only worth it for resource-heavy, time-sensitive model training where parallel computing is mandatory, but completely unjustifiable for routine prototyping or standard hosting where a cheap CPU instance works just as well.


#### JSON info regarding the estimates
{
	"Name": "My Estimate",
	"Total Cost": {
		"monthly": "491.26",
		"upfront": "0.00",
		"12 months": "5895.12"
	},
	"Metadata": {
		"Currency": "USD",
		"Locale": "en_US",
		"Created On": "11/24/2026",
		"Legal Disclaimer": "AWS Pricing Calculator provides only an estimate of your AWS fees and doesn't include any taxes that might apply. Your actual fees depend on a variety of factors, including your actual usage of AWS services.",
		"Share Url": "https://calculator.aws/#/estimate?id=a83d0ba15c688f67ff629357a5551c47cf9c1a85"
	},
	"Groups": {
		"Services": [
			{
				"Service Name": "Amazon EC2 ",
				"Description": "Scenario A ",
				"Region": "US East (N. Virginia)",
				"Status": "",
				"Service Cost": {
					"monthly": "1.66",
					"upfront": "0.00",
					"12 months": "19.97"
				},
				"Properties": {
					"Tenancy": "Shared Instances",
					"Operating system": "Linux",
					"Workload": "Consistent, Number of instances: 1",
					"Advance EC2 instance": "t3.micro",
					"Pricing strategy": "On-Demand Utilization: 160 Hours/Month",
					"Enable monitoring": "disabled",
					"DT Inbound: Not selected": "0 TB per month",
					"DT Outbound: Not selected": "0 TB per month",
					"DT Intra-Region:": "0 TB per month"
				}
			},
			{
				"Service Name": "Amazon EC2 ",
				"Description": "Scenario B",
				"Region": "US East (N. Virginia)",
				"Status": "",
				"Service Cost": {
					"monthly": "489.60",
					"upfront": "0.00",
					"12 months": "5875.20"
				},
				"Properties": {
					"Tenancy": "Shared Instances",
					"Operating system": "Linux",
					"Workload": "Consistent, Number of instances: 1",
					"Advance EC2 instance": "p3.2xlarge",
					"Pricing strategy": "On-Demand Utilization: 160 Hours/Month",
					"Enable monitoring": "disabled",
					"DT Inbound: Not selected": "0 TB per month",
					"DT Outbound: Not selected": "0 TB per month",
					"DT Intra-Region:": "0 TB per month"
				}
			}
		]
	}
} 