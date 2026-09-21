from agents import build_search_agent, build_reader_agent, writer_chain, critic_prompt

def run_research_pipeline(topic:str) -> dict:
    state = {}
    
    #search agent
    
    print("\n"+" ="*50)
    print("step 1: Search Agent is working...")
    print("\n"+" ="*50)
    
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user",f"Find recent,reliable and detailed information about topic: {topic}")]
    })
    
    state["search_result"] = search_result['messages'][-1]['content']
    
    print("\n search result: \n", state["search_result"])
    
    #reader agent
    
    print("\n"+" ="*50)
    print("step 2: Reader Agent is working...")
    print("\n"+" ="*50)
    
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    
    state["scraped_content"] = reader_result['messages'][-1]['content']  
    
    print("\n scraped content: \n", state["scraped_content"])
    
    #writer chain
    
    print("\n"+" ="*50)
    print("step 3: Writer Chain is working...")
    print("\n"+" ="*50)
    
    research_combined = (
        f"Search Results:\n{state['search_result']}\n\n"
        f"Scraped Content:\n{state['scraped_content']}" 
        
    ) 
    
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    
    print("\n Final report: \n", state["report"])
    
    #critic report
    
    print("\n"+" ="*50)
    print("step 4: Critic Chain is working...")
    print("\n"+" ="*50)
    
    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
     
    print("\n Critic feedback: \n", state["feedback"])
    
    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)
     