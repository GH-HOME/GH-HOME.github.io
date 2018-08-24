---
layout: page
title: Index of posts
permalink: /post_index/
---

<div class="posts">
<ul>
  {% for post in site.posts %}
    <li>
	<a style="color: black;" href="{{ site.baseurl | remove: '<p>' | remove: '</p>' }}{{ post.url | remove: '<p>' | remove: '</p>' }}">{{ post.title | remove: '<p>' | remove: '</p>' }}</a>
</li>
  {% endfor %}
</ul>
</div> 

