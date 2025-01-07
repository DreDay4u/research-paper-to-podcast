from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from dotenv import load_dotenv
import os
from tools import PodcastAudioGenerator, PodcastMixer, VoiceConfig

def setup_directories():
    """Set up organized directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dirs = {
        'BASE': f'outputs/{timestamp}',
        'SEGMENTS': f'outputs/{timestamp}/segments',  # Individual voice segments
        'FINAL': f'outputs/{timestamp}/podcast',      # Final podcast file
        'DATA': f'outputs/{timestamp}/data'          # Metadata/JSON files
    }
    
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    return dirs

# Load environment variables
load_dotenv()

# --- PDF Knowledge Source ---
research_paper = PDFKnowledgeSource(file_paths="long-context_vs_RAG.pdf")

# --- Pydantic Models definitions ---
class PaperSummary(BaseModel):
    """Summary of a research paper."""
    title: str = Field(..., description="Title of the research paper")                   
    main_findings: List[str] = Field(..., description="Key findings as a list of strings")
    methodology: str = Field(..., description="Research methodology as a single text block")
    key_implications: List[str] = Field(..., description="Implications as a list of strings")
    limitations: List[str] = Field(..., description="Limitations as a list of strings")
    future_work: List[str] = Field(..., description="Future research directions as a list")
    summary_date: datetime = Field(..., description="Timestamp of summary creation")

class DialogueLine(BaseModel):
    """Dialogue line for a podcast script."""
    speaker: str = Field(..., description="Name of the speaker (Julia or Guido)")
    text: str = Field(..., description="The actual dialogue line")

class PodcastScript(BaseModel):
    """Podcast script with dialogue lines."""
    dialogue: List[DialogueLine] = Field(..., description="Ordered list of dialogue lines")

class AudioGeneration(BaseModel):
    """Audio generation result with metadata."""
    segment_files: List[str] = Field(..., description="List of generated audio segment files")
    final_podcast: str = Field(..., description="Path to the final mixed podcast file")

# --- LLM Setup ---
summary_llm = LLM(
    model="openai/o1-preview",
    temperature=0.0,
)

script_llm = LLM(
    model="openai/o1-preview",
    temperature=0.3,
)

script_enhancer_llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    temperature=0.7,
)

audio_llm = LLM(
    model="cerebras/llama3.3-70b",
    temperature=0.0,
)

# Create and configure tools
dirs = setup_directories()
audio_generator = PodcastAudioGenerator(output_dir=dirs['SEGMENTS'])

# Julia: Enthusiastic expert
audio_generator.add_voice(
    "Julia", 
    os.getenv("CLAUDIA_VOICE_ID"),
    VoiceConfig(
        stability=0.35,  # More variation for natural enthusiasm
        similarity_boost=0.75,  # Maintain voice consistency
        style=0.65,  # Good expressiveness without being over the top
        use_speaker_boost=True
    )
)

# Guido: Engaged and curious
audio_generator.add_voice(
    "Guido", 
    os.getenv("BEN_VOICE_ID"),
    VoiceConfig(
        stability=0.4,  # Slightly more stable but still natural
        similarity_boost=0.75,
        style=0.6,  # Balanced expressiveness
        use_speaker_boost=True
    )
)

podcast_mixer = PodcastMixer(output_dir=dirs['FINAL'])


# --- Agents ---
researcher = Agent(
    role="Research Analyst",
    goal="Create comprehensive yet accessible research paper summaries",
    backstory="""You're a PhD researcher with a talent for breaking down complex
    academic papers into clear, understandable summaries. You excel at identifying
    key findings and their real-world implications.""",
    verbose=True,
    llm=summary_llm
)

script_writer = Agent(
    role="Podcast Script Writer",
    goal="Create engaging and educational podcast scripts about technical topics",
    backstory="""You're a skilled podcast writer who specializes in making technical 
    content engaging and accessible. You create natural dialogue between two hosts: 
    Julia (a knowledgeable expert who explains concepts clearly) and Guido (an informed 
    co-host who asks thoughtful questions and helps guide the discussion).""",
    verbose=True,
    llm=script_llm
)

script_enhancer = Agent(
    role="Podcast Script Enhancer",
    goal="Enhance podcast scripts to be more engaging while maintaining educational value",
    backstory="""You're a veteran podcast producer who specializes in making technical 
    content both entertaining and informative. You excel at adding natural humor, 
    relatable analogies, and engaging banter while ensuring the core technical content 
    remains accurate and valuable. You've worked on shows like Lex Fridman's podcast, 
    Hardcore History, and the Joe Rogan Experience, bringing their signature blend of 
    entertainment and education.""",
    verbose=True,
    llm=script_enhancer_llm 
)

audio_generator_agent = Agent(
    role="Audio Generation Specialist",
    goal="Generate high-quality podcast audio with natural-sounding voices",
    backstory="""You are an expert in audio generation and processing. You understand 
    how to generate natural-sounding voices and create professional podcast audio. You 
    consider pacing, tone, and audio quality in your productions.""",
    verbose=True,
    allow_delegation=False,
    tools=[audio_generator, podcast_mixer],
    llm=audio_llm
)

# --- Tasks ---
summary_task = Task(
    description="""Read and analyze the provided research paper: {paper}.
    
    Create a comprehensive summary that includes:
    1. Main findings and conclusions
    2. Methodology overview
    3. Key implications for the field
    4. Limitations of the study
    5. Suggested future research directions
    
    Make the summary accessible to an educated general audience while maintaining accuracy.""",
    expected_output="A structured summary of the research paper with all key components.",
    agent=researcher,
    output_pydantic=PaperSummary,
    output_file="output/metadata/paper_summary.json"
)

podcast_task = Task(
    description="""Using the paper summary, create an engaging and informative podcast conversation 
    between Julia and Guido about the research. Make it feel like a natural, enjoyable conversation 
    between two tech enthusiasts who genuinely love their field - similar to the Joe Rogan Experience 
    but focused on tech.

    Host Dynamics:
    - Julia: A knowledgeable but relatable expert who:
        • Explains technical concepts with enthusiasm
        • Shares relevant personal experiences with AI and tech
        • Can connect the research to broader tech trends
        • Uses casual expressions and shows genuine excitement
    
    - Guido: An engaged and curious co-host who:
        • Asks insightful questions and follows interesting threads
        • Brings up related examples from other tech domains
        • Helps make connections to practical applications
        • Naturally guides the conversation back to main topics

    Conversation Flow:
    1. Core Discussion: Focus on the RULER research and findings
    2. Natural Tangents (examples):
        • Personal experiences with AI language models
        • Similar problems in other areas of tech
        • Funny or interesting AI interaction stories
        • Related developments in the tech industry
        • Practical implications for developers or users
    3. Smooth Returns: Natural ways to bring the conversation back:
        • "You know, this actually relates to what we were discussing about RULER..."
        • "That reminds me of how the researchers approached this problem..."
        • "Speaking of which, this connects perfectly with the paper's findings..."

    Example Flow:
    Julia: "The way RULER handles length control is fascinating..."
    Guido: "Oh man, this reminds me of when GitHub Copilot generated this massive chunk of code 
    when I just wanted a simple function. It was like asking for a screwdriver and getting an 
    entire hardware store!"
    Julia: "That's hilarious! And you know what's interesting? That's exactly the kind of 
    problem these researchers were trying to solve with RULER..."

    Writing Guidelines:
    1. Allow natural divergence into relevant topics
    2. Use personal anecdotes and examples
    3. Connect tangents back to main discussion smoothly
    4. Keep technical content accurate but conversational
    5. Maintain engagement through relatable stories
    
    Note: Convey reactions through natural language rather than explicit markers like *laughs*.""",
    expected_output="A well-balanced podcast script that combines technical content with engaging tangents.",
    agent=script_writer,
    context=[summary_task],
    output_pydantic=PodcastScript,
    output_file="output/metadata/podcast_script.json"
)

enhance_script_task = Task(
    description="""Take the initial podcast script and enhance it to be more engaging 
    and conversational while maintaining its educational value.
    
    IMPORTANT RULES:
    1. NEVER change the host names - always keep Julia and Guido exactly as they are
    2. NEVER add explicit reaction markers like *chuckles*, *laughs*, etc.
    3. NEVER add new hosts or characters
    
    Enhancement Guidelines:
    1. Add Natural Elements:
        • Include natural verbal reactions ("Oh that's fascinating", "Wow", etc.)
        • Keep all dialogue between Julia and Guido only
        • Add relevant personal anecdotes or examples that fit their established roles:
            - Julia as the knowledgeable expert
            - Guido as the engaged and curious co-host
        • Express reactions through words rather than action markers
    
    2. Improve Flow:
        • Ensure smooth transitions between topics
        • Add brief casual exchanges that feel natural
        • Include moments of reflection or connection-making
        • Balance technical depth with accessibility
    
    3. Maintain Quality:
        • Keep all technical information accurate
        • Ensure added content supports rather than distracts
        • Preserve the core findings and insights
        • Keep the overall length reasonable
    
    4. Add Engagement Techniques:
        • Include thought-provoking analogies by both hosts
        • Add relatable real-world examples
        • Express enthusiasm through natural dialogue
        • Include collaborative problem-solving moments
        • Inject humor where appropriate and it has to be funny

    Natural Reaction Examples:
    ✓ RIGHT: "Oh, that's fascinating!"
    ✓ RIGHT: "Wait, that doesn't make sense!"
    ✓ RIGHT: "Wait, really? I hadn't thought of it that way."
    ✓ RIGHT: "That's such a great point."
    ✗ WRONG: *chuckles* or *laughs* or any other action markers
    ✗ WRONG: Adding new speakers or changing host names
    
    The goal is to make the content feel like a conversation between Julia and Guido
    who are genuinely excited about the topic, while ensuring listeners learn 
    something valuable.""",
    expected_output="An enhanced version of the podcast script that's more engaging and natural",
    agent=script_enhancer,
    context=[summary_task, podcast_task],
    output_pydantic=PodcastScript,
    output_file="output/metadata/enhanced_podcast_script.json"
)

audio_task = Task(
    description="""Generate high-quality audio for the podcast script and create the final podcast.
    
    The script will be provided in the context as a list of dialogue entries, each with:
    - speaker: Either "Julia" or "Guido"
    - text: The line to be spoken
    
    Process:
    1. Generate natural-sounding audio for each line of dialogue using appropriate voices
    2. Apply audio processing for professional quality:
       - Normalize audio levels
       - Add subtle fade effects between segments
       - Apply appropriate pacing and pauses
    3. Mix all segments into a cohesive final podcast
    
    Voice Assignments:
    - For Julia's lines: Use configured Julia voice
    - For Guido's lines: Use configured Guido voice
    
    Quality Guidelines:
    - Ensure consistent audio levels across all segments
    - Maintain natural pacing and flow
    - Create smooth transitions between speakers
    - Verify audio clarity and quality""",
    expected_output="A professional-quality podcast audio file with natural-sounding voices and smooth transitions",
    agent=audio_generator_agent,
    context=[enhance_script_task],
    output_pydantic=AudioGeneration,
    output_file="output/metadata/audio_generation_meta.json"
)

# --- Crew and Process ---
crew = Crew(
    agents=[researcher, script_writer, script_enhancer, audio_generator_agent],
    tasks=[summary_task, podcast_task, enhance_script_task, audio_task],
    process=Process.sequential,
    knowledge_sources=[research_paper],
    verbose=True
)

if __name__ == "__main__":    
    # Update task output files
    summary_task.output_file = os.path.join(dirs['DATA'], "paper_summary.json")
    podcast_task.output_file = os.path.join(dirs['DATA'], "podcast_script.json")
    enhance_script_task.output_file = os.path.join(dirs['DATA'], "enhanced_podcast_script.json")
    audio_task.output_file = os.path.join(dirs['DATA'], "audio_generation_meta.json")
    
    # Run the podcast generation process
    results = crew.kickoff(inputs={"paper": "long-context_vs_RAG.pdff"})