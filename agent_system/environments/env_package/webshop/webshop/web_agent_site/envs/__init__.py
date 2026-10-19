from gym.envs.registration import register

def __getattr__(name):
    if name == 'WebAgentSiteEnv':
        from .web_agent_site_env import WebAgentSiteEnv
        return WebAgentSiteEnv
    if name == 'WebAgentTextEnv':
        from .web_agent_text_env import WebAgentTextEnv
        return WebAgentTextEnv
    raise AttributeError(name)

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)
