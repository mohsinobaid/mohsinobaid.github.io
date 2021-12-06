---
layout: post
title:  "Divergence-Measures-Breakdown"
date:   2021-05-25 04:27:59 -0400
categories: jekyll update
---

The Papers that I am reading:

  Divergence Measures

    Divergence measures are useful in statistics, information theory and lately in Machine Learning. (for reasons i will get into shortly).

    Ok maybe right away...
    In Machine Learning, Minimizing the Divergence between a probability distribution that you want to learn and the input, will result in learning.

    Unfortunately, calculating the real value of the Mutual Information is not feasible in the high dimensional space of deep learning models.

    So, people have been relaxing the constraints of a real mutual information measurement in order to create new measurements that are 'good enough'.

    You can even estimate the gradients of the MI using different measures than the ones you are using for the values.?

    In mathematical terms, this is called a lower bound.?




Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
