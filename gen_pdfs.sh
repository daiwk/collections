#echo "本文地址[https://github.com/daiwk/collections/blob/master/pdfs/collections.pdf](https://github.com/daiwk/collections/blob/master/pdfs/collections.pdf)\n" > collections.md

#cat ./posts/full.md | python3 trans_format.py >> ./pdfs/collections-pdf.md


python3 gen_dot_sub.py


function change_format()
{
    raw_name=$1
    out_name=$2
    index=$3
    final_out_name=${index}.${out_name}
    cat ./posts/$raw_name.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/${final_out_name}-pdf.md
    cat ./posts/$raw_name.md.raw | python3 ./posts/change_format_md.py > ./posts/${final_out_name}.md
    cd pdfs
    pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./${final_out_name}-pdf.md -o ./${out_name}.pdf
    cd -
}

change_format pre_llm pre_llm 1.1
change_format llm_intro llm_intro 1.2
change_format llm_archs llm_archs 1.3
change_format llm_sft_and_usages llm_sft_and_usages 1.4
change_format llm_alignment llm_alignment 1.5
change_format llm_multimodal llm_multimodal 1.6
change_format llm_recommend llm_recommend 1.7
change_format llm_o1 llm_o1 1.8

cat ./posts/pre.md  \
        ./posts/pre_llm.md.raw \
        ./posts/llm_intro.md.raw \
        ./posts/llm_archs.md.raw \
        ./posts/llm_sft_and_usages.md.raw \
        ./posts/llm_alignment.md.raw \
        ./posts/llm_multimodal.md.raw \
        ./posts/llm_recommend.md.raw \
        ./posts/llm_o1.md.raw \
        ./posts/llm_others.md.raw \
> ./posts/llm_aigc.md.raw

change_format llm_aigc llm_aigc 1

change_format recommend recommend 2
change_format full collections 9
change_format int-ml int-ml 8


#cat ./posts/recommend.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/2.recommend-pdf.md
#cat ./posts/recommend.md.raw | python3 ./posts/change_format_md.py > ./posts/2.recommend.md
#
#cat ./posts/full.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/9.collections-pdf.md
#cat ./posts/full.md.raw | python3 ./posts/change_format_md.py > ./posts/9.collections.md
#
#cat ./posts/int-ml.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/8.int-ml-pdf.md
#cat ./posts/int-ml.md.raw | python3 ./posts/change_format_md.py > ./posts/8.int-ml.md


##pandoc -N -s --toc --smart  --pdf-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in -f markdown+markdown_in_html_blocks+raw_html-implicit_figures ./collections-pdf.md -o ./pdfs/collections.pdf


#pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./1.llm_aigc-pdf.md -o ./llm_aigc.pdf
#pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./2.recommend-pdf.md -o ./recommend.pdf
#pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./8.int-ml-pdf.md -o ./int-ml.pdf
#pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./9.collections-pdf.md -o ./collections.pdf

rm -rf pdfs/*.md

git add ./assets
git add posts
git add pdfs
git add gen_pdfs.sh
git add gen_dot_sub.py
git commit -m 'x'
git push
