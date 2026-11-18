# 참고문헌 전체 검증 및 본문 연결 기록

15개 문헌 모두의 존재·서지·주장 연결을 다시 확인했습니다. 각 항목에 실제 확인 범위를 적었으며, 초록·메타데이터 확인을 전문 전체 검토로 표현하지 않았습니다. 접근 제한이 있는 링크는 삭제하거나 가짜 링크로 판단하지 않고 저자 원고와 공식 기록으로 교차 확인했습니다.

## Aljundi et al. (2019)
Aljundi, R., Caccia, L., Belilovsky, E., Caccia, M., Lin, M., Charlin, L., & Tuytelaars, T. (2019). Online continual learning with maximally interfered retrieval. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), Advances in neural information processing systems (Vol. 32, pp. 11849–11860). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html

- 확인 범위: Published PDF title page, abstract, and Sections 1–3
- 근거 링크: https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf
- 뒷받침하는 내용: MIR selects replay memories using predicted interference from an upcoming update.
- 원고 적용: Related work 2.1; distinguishes replay selection from observational risk measurement.
- 해석 범위 제한: PER neither predicts a virtual parameter update nor selects replay examples. MIR is not an implemented baseline.
- 서지 주의점: PDF author order and Maximally title retained; proceedings metadata differs.

## Efron and Tibshirani (1993)
Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall. https://doi.org/10.1007/978-1-4899-4541-9

- 확인 범위: Book excerpt Chapters 1–2; publisher preview and DOI metadata corroborated
- 근거 링크: https://www.cns.nyu.edu/~eero/math-tools14/Handouts/Efron-ch1%2B2.pdf
- 뒷받침하는 내용: Bootstrap is a resampling framework for estimating sampling uncertainty.
- 원고 적용: Methods 3.12; supports resampling, not the adequacy of this experiment.
- 해석 범위 제한: Fixed-OOF resampling does not include refitting or new-seed variation.
- 서지 주의점: Original 1993 edition retained; full-book reading not claimed; publisher direct access blocked.

## Elazar et al. (2021)
Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schütze, H., & Goldberg, Y. (2021). Measuring and improving consistency in pretrained language models. Transactions of the Association for Computational Linguistics, 9, 1012–1031. https://doi.org/10.1162/tacl_a_00410

- 확인 범위: Official ACL metadata and abstract
- 근거 링크: https://aclanthology.org/2021.tacl-1.60/
- 뒷받침하는 내용: ParaRel evaluates consistency across paraphrased factual queries.
- 원고 적용: Related work 2.3; motivates sensitivity to input variation.
- 해석 범위 제한: Factual-query consistency is not label-referenced PER or later-task error prediction.
- 서지 주의점: Authors, title, year, pages and DOI match official record.

## Hu et al. (2022)
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9

- 확인 범위: Author preprint abstract and author list; ICLR record URL checked
- 근거 링크: https://arxiv.org/abs/2106.09685
- 뒷받침하는 내용: LoRA freezes pretrained weights and learns low-rank updates.
- 원고 적용: Related work 2.4 and Methods 3.6; adaptation mechanism.
- 해석 범위 제한: LoRA is not evidence that this study prevents forgetting; no full-fine-tuning comparison was run.
- 서지 주의점: ICLR 2022 retained rather than replacing it with preprint year 2021; OpenReview verification page limits direct access.

## Kirkpatrick et al. (2017)
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

- 확인 범위: Author record, abstract and linked PNAS DOI
- 근거 링크: https://arxiv.org/abs/1612.00796
- 뒷받침하는 내용: EWC slows updates to parameters important for previous tasks.
- 원고 적용: Related work 2.1; parameter regularization as a mitigation contrast.
- 해석 범위 제한: No EWC arm or mitigation comparison exists in the current experiment.
- 서지 주의점: Journal DOI preserved; publisher direct access blocked, author manuscript available.

## Li and Hoiem (2016)
Li, Z., & Hoiem, D. (2016). Learning without forgetting. In B. Leibe, J. Matas, N. Sebe, & M. Welling (Eds.), Computer vision – ECCV 2016 (pp. 614–629). Springer. https://doi.org/10.1007/978-3-319-46493-0_37

- 확인 범위: Author record and abstract; ECCV identity corroborated
- 근거 링크: https://arxiv.org/abs/1606.09282v1
- 뒷받침하는 내용: Learning without Forgetting uses new-task data while preserving old-task responses.
- 원고 적용: Related work 2.1; distillation-based retention context.
- 해석 범위 제한: The PER experiment does not use a distillation objective or compare LwF performance.
- 서지 주의점: ECCV 2016 conference version retained, not later journal version.

## Luo et al. (2023)
Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2023). An empirical study of catastrophic forgetting in large language models during continual fine-tuning [Preprint]. arXiv. https://arxiv.org/abs/2308.08747v1

- 확인 범위: Exact 2023 author record and abstract
- 근거 링크: https://arxiv.org/abs/2308.08747v1
- 뒷받침하는 내용: Studies retention of knowledge and abilities during continual LLM fine-tuning.
- 원고 적용: Related work 2.1; LLM-level empirical retention context.
- 해석 범위 제한: Its model-scale and task findings are not assumed to apply to this Qwen-LoRA setup.
- 서지 주의점: 2023 v1 retained; later 2025 revisions not silently substituted.

## Ribeiro et al. (2020)
Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 4902–4912). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.442

- 확인 범위: Official ACL metadata and abstract
- 근거 링크: https://aclanthology.org/2020.acl-main.442/
- 뒷받침하는 내용: CheckList tests NLP behavior using capabilities and test types, including invariance.
- 원고 적용: Related work 2.3; motivates transformation-based evaluation.
- 해석 범위 제한: A generated rewrite is not automatically a valid invariance test; semantic validity remains unconfirmed for the full bank.
- 서지 주의점: Authors, venue, pages and DOI match official record.

## Saito and Rehmsmeier (2015)
Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432

- 확인 범위: Publisher article abstract, metadata and discussion of imbalance
- 근거 링크: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
- 뒷받침하는 내용: Precision-recall evaluation is informative when positives are rare.
- 원고 적용: Related work 2.4 and Methods 3.9; explains AP alongside AUROC.
- 해석 범위 제한: This does not establish AP superiority in every setting or a universal threshold.
- 서지 주의점: Publisher author list, article number, year and DOI verified.

## Scialom et al. (2022)
Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In Y. Goldberg, Z. Kozareva, & Y. Zhang (Eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410

- 확인 범위: Official ACL metadata and abstract
- 근거 링크: https://aclanthology.org/2022.emnlp-main.410/
- 뒷받침하는 내용: Continual-T0 extends instruction learning across new tasks while evaluating retention.
- 원고 적용: Related work 2.1; distinct continual-learning setting.
- 해석 범위 제한: CT0 and its rehearsal setting are not direct baseline results for the current unrehearsed three-task run.
- 서지 주의점: Authors, year, pages and DOI verified.

## Shi et al. (2025)
Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., Wang, Z., Ebrahimi, S., & Wang, H. (2025). Continual learning of large language models: A comprehensive survey. ACM Computing Surveys, 58(5), Article 120. https://doi.org/10.1145/3735633

- 확인 범위: Publisher search-index text and author arXiv record
- 근거 링크: https://doi.org/10.1145/3735633
- 뒷받침하는 내용: Survey distinguishes continual pretraining, domain adaptation and continual fine-tuning.
- 원고 적용: Introduction and Related work 2.1; scope of the study.
- 해석 범위 제한: A survey does not validate the PER hypothesis or this design.
- 서지 주의점: Publisher publication date is November 20, 2025; issue 58(5) is associated with 2026. Publication year 2025 retained and ambiguity documented; direct fetch returns 403.

## Swayamdipta et al. (2020)
Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., & Choi, Y. (2020). Dataset cartography: Mapping and diagnosing datasets with training dynamics. In B. Webber, T. Cohn, Y. He, & Y. Liu (Eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 9275–9293). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.746

- 확인 범위: Official ACL metadata and abstract
- 근거 링크: https://aclanthology.org/2020.emnlp-main.746/
- 뒷받침하는 내용: Dataset cartography uses true-label confidence and its variability over training.
- 원고 적용: Related work 2.2; motivates a strong uncertainty comparison.
- 해석 범위 제한: A single Task-A maximum candidate probability is not their full training-dynamics measure.
- 서지 주의점: Authors, year, pages and DOI verified.

## Toneva et al. (2019)
Toneva, M., Sordoni, A., Tachet des Combes, R., Trischler, A., Bengio, Y., & Gordon, G. J. (2019). An empirical study of example forgetting during deep neural network learning. International Conference on Learning Representations. https://openreview.net/forum?id=BJlxm30cKm

- 확인 범위: Author-hosted ICLR PDF, Sections 1, 3 and 4
- 근거 링크: https://www.cs.cmu.edu/~mktoneva/papers/iclr2019_sample_forgetting_arXiv.pdf
- 뒷받침하는 내용: Tracks example forgetting as correct-to-incorrect transitions across presentations during single-task training.
- 원고 적용: Related work 2.2 and Discussion; distinguishes transition outcomes.
- 해석 범위 제한: Current outcomes are held-out examples at two later task checkpoints, not event counts over training updates or proof of irreversible loss.
- 서지 주의점: Exact title and author order verified; OpenReview redirects to verification, author-hosted PDF resolves.

## Yang et al. (2024)
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). Qwen2.5 technical report [Preprint]. arXiv. https://arxiv.org/abs/2412.15115v1

- 확인 범위: Exact v1 author record and abstract
- 근거 링크: https://arxiv.org/abs/2412.15115v1
- 뒷받침하는 내용: Qwen2.5 report describes the base and instruction-tuned model family.
- 원고 적용: Related work 2.4 and Methods 3.6; model provenance.
- 해석 범위 제한: Family-level benchmark findings are not evidence for this 1.5B continual-learning outcome.
- 서지 주의점: 2024 v1 and APA long-author form retained; model repository revision separately pinned.

## Zhang et al. (2015)
Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, & R. Garnett (Eds.), Advances in neural information processing systems (Vol. 28, pp. 649–657). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html

- 확인 범위: Official proceedings search-index PDF and author arXiv record
- 근거 링크: https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf
- 뒷받침하는 내용: Constructs text-classification benchmarks including AG News, Yelp and DBpedia.
- 원고 적용: Related work 2.4 and Methods 3.7; benchmark provenance.
- 해석 범위 제한: Its classifier architecture and reported scores are not this experiment; the current Hugging Face revisions define exact inputs.
- 서지 주의점: Official title, authors and proceedings identity verified; direct fetch intermittently times out.
