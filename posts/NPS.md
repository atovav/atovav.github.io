---
layout: template1
title: NPS analysis
---

Monthly NPS
================
Alan<br />
2021<br />

<p>The following is a brief analysis of a Monthly NPS presentation. The company and slides are omitted, and the values are modified for confidentiality.<br />
The first thing was to graph the NPS and the total number of responses over time. All the graphs were made in tableau for dashboard purposes and then send as image to a presentation.</p>

<h4>Overall NPS</h4>
<p>We see that the NPS is largely stable, in November we recovered from the problems in October.<br />
In total responses, we continue to increase the number of responses</p>

<div class="bigcenterimgcontainer">
<img src="img/npstime.jpg" alt style>
</div>

<h4>NPS by country</h4>
<p>Overall NPS shows an improvement, however that in not the complete story. If we see the changes in NPS by country, then we see a different story.<br />
We see 4 countries with improvements and 2 countries with decreases. The important one and the problem is PA. PA has a 22% decrease to the previous month, so we need to focus in that area.</p>

<div class="bigcenterimgcontainer">
<img src="img/npsbycountry.jpg" alt style>
</div>

<h4>PA: NPS components breakdown – Passives decreasing.</h4>
<p>We are going to breakdown NPS to see where the problem lies. <br />
First, we see that the number of passives is decreasing. The percentage decrease was 86% from the previous month.</p>

<div class="bigcenterimgcontainer">
<img src="img/panpspass.jpg" alt style>
</div>

<h4>PA: NPS components breakdown – Promoters decreasing.</h4>

<p>One problem that we found is the number of promoters decreased by 4%!</p>

<div class="bigcenterimgcontainer">
<img src="img/panpsprom.jpg" alt style>
</div>

<h4>PA: NPS components breakdown – Detractors increasing.</h4>

<p>We saw decreases in passives and promoters. Leading to a sharp increase in detractors. This is the reason why PA has a 22% decrease from the previous month.</p>

<div class="bigcenterimgcontainer">
<img src="img/panspdes1.jpg" alt style>
</div>

<h4>PA: NPS components breakdown – Detractors more than double</h4>

<p>The increase of detractors from the previous month is 236%.  We need to find the cause of this sharp increase. </p>

<div class="bigcenterimgcontainer">
<img src="img/panspdes2.jpg" alt style>
</div>

<p>We can find what could be the problem. The data has a column for issues; however, we found a problem in this specific case.  </p>

<h4>Issue tag is irrelevant for this country</h4>

<p>Of all the countries, PA is the one that does not have any issue tag. <br />
Recommendation: See why this country does not have any issue tag.</p>


<table class="tg">
<colgroup>
<col style="width: 80px">
<col style="width: 80px">
<col style="width: 80px">
<col style="width: 80px">
<col style="width: 80px">
<col style="width: 80px">
<col style="width: 80px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">CL</th>
    <th class="tg-0lax">CO</th>
    <th class="tg-0lax">MX</th>
    <th class="tg-0lax">PA</th>
    <th class="tg-0lax">PE</th>
    <th class="tg-0lax">VE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9s94">Nulls</td>
    <td class="tg-pb6s">30</td>
    <td class="tg-pb6s">353</td>
    <td class="tg-0lax"></td>
    <td class="tg-i99s">68</td>
    <td class="tg-pb6s">63</td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-9s94">Reasons</td>
    <td class="tg-pb6s">10</td>
    <td class="tg-pb6s">1,806</td>
    <td class="tg-pb6s">2,704</td>
    <td class="tg-0lax"></td>
    <td class="tg-pb6s">821</td>
    <td class="tg-pb6s">350</td>
  </tr>
</tbody>
</table>
<br/>

<h4>Using Comments as a guide</h4>

<p>Another way to understand why there is an increase in detractors is to look at the comments. <br />
We took each comment divided into words and assigned a neutral, negative, and positive label. We removed the neutral label and graph the words. Blue means positive and red means negative. </p>

<div class="bigcenterimgcontainer">
<img src="img/wordcom.jpg" alt style>
</div>

<p>We see that most of the positive comments come from service, while most of the complains come from delivery, time, damage. <br />
We have now a place to start. </p>

<h4>Final Recommendations</h4>
1)	Find why PA is the only country without issues tag. <br />
2)	There is an issue with deliveries. Find why this specific delivery type is failing. <br />
<br />

