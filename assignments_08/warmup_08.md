# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1
### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core economic model of cloud computing is pay-as-you-go rental, where you rent computing resources (storage, processing power, networking) and pay only for what you actually use. It differs from owning physical servers because you avoid the heavy upfront expense and ongoing maintenance of physical hardware.

## Cloud Concepts Question 2
### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means making a single machine bigger by upgrading its hardware (more CPU, more RAM, or a larger GPU) — for example, upgrading an instance to fit a massive machine learning model during local training. Horizontal scaling means adding more machines and splitting the workload across them — for example, launching multiple web server instances behind a load balancer to absorb sudden traffic surges.

### Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

* A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
    Horizontal scaling applies because stateless web traffic can easily be split across multiple newly provisioned server instances and wound back down when traffic subsides.
* A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
    Vertical scaling applies because a single model training process typically requires a more powerful individual machine with higher GPU compute and memory capacity.
* A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
    Horizontal scaling applies because the workload is explicitly partitionable across multiple separate worker machines running in parallel.


## Cloud Concepts Question 3

### Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

* Gmail: SaaS — It is a ready-to-use end-user application running in the browser where the provider handles all underlying software and servers.
* Azure Virtual Machines: IaaS — It provides raw virtualized compute infrastructure where you configure the operating system, runtime, and software yourself.
* AWS S3 (Simple Storage Service): IaaS — It offers raw cloud storage capacity accessible over the network without managing an underlying execution platform.
* GitHub Codespaces: PaaS — It provides a managed development environment and platform where the underlying hardware and OS are handled, and you focus on writing code and environment configs.
* Snowflake: SaaS / Managed Data Platform — It is a specialized, fully managed data warehousing platform running on top of cloud infrastructure that abstracts database administration away from the user.
* Supabase: BaaS — It delivers a pre-packaged backend platform (hosted PostgreSQL, authentication, and auto-generated APIs) so you don't build or manage server infrastructure from scratch.

### Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

* IaaS (Infrastructure as a Service)
    Description: The provider rents you raw computing resources (virtual machines, storage, networking).
    Example: Azure Virtual Machines (or AWS EC2).
    Developer responsibility: Managing everything from the operating system up — choosing and configuring the OS, installing software, patching, environment settings, and application code.
* PaaS (Platform as a Service)
    Description: The provider manages the underlying infrastructure, operating system, and hardware health, providing a platform to run applications.
    Example: Azure App Service (or AWS Elastic Beanstalk).
    Developer responsibility: Bringing and managing your own code and application configuration.
* SaaS (Software as a Service)
    Description: Complete applications built, hosted, and maintained by a third-party provider that you access directly via a browser.
    Example: Gmail (or Google Docs).
    Developer responsibility: Almost nothing — you only manage your account access, user data, and application settings.


## Cloud Concepts Question 4

### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

Cloud providers (AWS, GCP, Azure) provide the full raw toolkit (compute, storage, databases) where you must manually assemble and configure your own architecture. In contrast, managed data platforms (Databricks, Snowflake) sit as a curated layer on top of those clouds, pre-wiring and optimizing the infrastructure specifically for data processing and analytics.   
* What you gain: Speed and simplicity — it is much faster to get started with large-scale data workflows without spending significant time configuring low-level infrastructure.
* What you give up: Flexibility (fine-grained control over underlying components) and potentially higher costs compared to managing raw cloud resources directly.

## Cloud Concepts Question 5
### The lesson describes a situation where the cloud is probably not the right choice. What is that situation?

The cloud is probably not the right choice when your dataset fits comfortably on a single machine and you do not have massive compute demands (such as building an initial local prototype), where running locally on your own computer is significantly faster, simpler, and cheaper than dealing with cloud overhead and costs.


# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

* Amazon Web Services (AWS): The oldest, largest, and pioneer of cloud infrastructure; its primary strength is having the broadest and most mature service catalog across enterprise, startup, and nonprofit sectors.
* Google Cloud Platform (GCP): Best known for its strength in data and machine learning, leveraging Google's heritage in pioneering modern distributed systems (such as MapReduce, Bigtable, and BigQuery).
* Microsoft Azure: The dominant player in enterprise and government environments due to its seamless, deep integration with Windows, Active Directory, and Microsoft 365 ecosystems.


## Cloud Landscape Question 2

### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

1. Access: Supabase enables instant, frictionless self-provisioning in under two minutes with a free tier, completely avoiding Azure's cumbersome enterprise tenant invitations and permission bottlenecks.
2. Pedagogical fit: Working directly with structured relational data (rows and columns in PostgreSQL) teaches broadly transferable SQL and data modeling skills, unlike storing opaque files by path in Azure Blob Storage.
3. Pipeline coherence: The course ETL stages cleanly map to two inspectable, linked relational tables (weather_raw and weather_enriched), making debugging and verifying data transformations straightforward.


### Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

When starting a new project, evaluate tools based on how quickly they eliminate setup friction and how naturally their data model aligns with your actual problem, rather than defaulting to the largest, most feature-heavy enterprise platform.

## Cloud Landscape Question 3

### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

* You need to store 10 TB of image files and retrieve them by filename from any machine.
    Category: Object storage
    Provider / Product: AWS S3 (or Azure Blob Storage / Google Cloud Storage / Cloudflare R2)
* You need to run an ML training job on a GPU for four hours, then shut it down.
    Category: ML platform (or Compute)
    Provider / Product: AWS SageMaker (or GCP Vertex AI / Azure ML)
* You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
    Category: Serverless compute
    Provider / Product: GCP Cloud Run (or AWS Lambda / Azure Functions)
* You need to send structured data to a large language model and get a text response back.
    Category: LLM API
    Provider / Product: Azure OpenAI Service (or AWS Bedrock / GCP Vertex AI)


## Cloud Landscape Question 4
### The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

Project:
A daily weather reporting app that collects regional weather metrics, enriches the forecasts with automated natural language summaries, and delivers them via a dashboard.

Storage & Database: Supabase (Managed relational DB) to store raw weather observations and enriched forecasts.
Enrichment: Azure OpenAI (LLM API) to generate concise text summaries from tabular metrics.
Hosting / Edge: Cloudflare Workers (Serverless compute) to serve the lightweight web frontend.

Consolidation Tradeoffs:
Consolidating onto a single provider (e.g., AWS or Azure) simplifies billing, unifies identity and access management (IAM), and eliminates cross-cloud networking/egress fees. However, you give up developer simplicity and "best-of-breed" tooling, forcing you to manage significantly more complex infrastructure configurations for components that other platforms handle out of the box.