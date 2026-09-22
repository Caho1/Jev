const tabs=[['community.html','全部概览'],['community-easy.html','基础判断'],['community-standard.html','路由与规则'],['community-hard.html','困难决策'],['community-reviews.html','评论分类'],['community-phishing.html','钓鱼邮件']];
const nav=document.createElement('nav');nav.className='test-tabs';nav.setAttribute('aria-label','测试页面');
const current=location.pathname.split('/').pop();for(const [path,label] of tabs){const a=document.createElement('a');a.href=path;a.textContent=label;if(path===current)a.setAttribute('aria-current','page');nav.append(a);}document.querySelector('header').after(nav);
