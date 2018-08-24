---
layout: page
title: Blog
permalink: /blog/
---


<div class="posts">
  {% for post in site.posts limit:10 %}
    <article class="post">
	<h3 style="color: #084B8A; font-size:130%; font-weight:normal;"><a href="{{ site.baseurl | remove: '<p>' | remove: '</p>' }}{{ post.url | remove: '<p>' | remove: '</p>' }}">{{ post.title | remove: '<p>' | remove: '</p>' }}</a></h3> 

	<span style="color: #A0A0A0; font-size:90%;">{{ post.date  | date_to_string  | remove: '<p>' | remove: '</p>' }} {% if post.comments %}(<a style="color: #A4A4A4;" href="{{ site.baseurl }}{{ post.url }}#disqus_thread">0 Comments</a>){% endif %}</span>  
    </article>
  {% endfor %}
<a href="{{ site.baseurl | remove: '<p>' | remove: '</p>' }}/post_index">>> Check all the posts here</a>
</div> 

