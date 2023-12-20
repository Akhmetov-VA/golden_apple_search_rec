from __future__ import annotations

from os import listdir

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


API_HOST = "api"
API_PORT = 8080


def show_theory_block():
	st.markdown(
		"""
#### Актуальность тематики

Последнее время становится все более популярным создание систем, которые
способны угадывать предпочтения и нужды пользователей и на основе этого предлагать
подходящие решения. Крупные компании такие, как Amazon, Apple, eBay, Pandora и т.д., используют
рекомендательные системы в составе своих сервисов.

Задача рекомендательной системы – проинформировать пользователя о товаре, который ему может быть 
наиболее интересен в данный момент времени. Клиент получает информацию, а сервис зарабатывает на 
предоставлении качественных услуг. Услуги — это не обязательно прямые продажи предлагаемого товара. 
Сервис также может зарабатывать на комиссионных или просто увеличивать лояльность пользователей,
которая потом выливается в рекламные и иные доходы. В зависимости от модели бизнеса рекомендации
могут быть его основой, как, например, у TripAdvisor, а могут быть просто удобным дополнительным 
сервисом (как, например, в каком-нибудь интернет-магазине одежды), призванным улучшить 
Customer Experience и сделать навигацию по каталогу более удобной.

Персонализация онлайн-маркетинга – очевидный тренд последнего десятилетия

Пока вы изучаете ассортимент, специальный алгоритм собирает досье: какие цвета и модели вам нравятся, 
что покупаете, а что удаляете из корзины. Программа находит похожие товары и в первую очередь показывает 
вам именно их — как если бы в ассортименте магазина были только ваши любимые вещи.

Рекомендательные системы приносят очевидную выгоду владельцам онлайн-магазинов, различных сервисов и приложений. 
Они показывают пользователю то, что ему интересно, и генерируют прибыль.

#### Кому будет полезен этот кейс?

- Аналитикам данных
- Бизнес-информатикам
- Бизнес-аналитикам
- Маркетологам
- Директорам по данным (CDO - Chief Data Officer)
- Руководителя по цифровой трансформации (CDO – Chief Digital Officer)
- Научным сотрудникам

#### А если у меня другой профиль?

Во-первых, всегда полезно знать о современных технологиях, которые применяют в своей работе
лидеры рынков.

Во-вторых, в ходе выполнения работы вы сможете изучить и понять этапы разработки рекомендательных
систем и применить их в своём профиле.

#### Цели и задачи

**Цель данной лабораторной работы** - ознакомить студентов с практикой применения искусственного интеллекта 
для рекомендательных систем.

**Задачи**:

1) Ознакомиться с теоретическими аспектами применения машинного обучения в рекомендательных системах;
2) Попробовать на практике одну из рекомендательных систем, реализованную методами машинного обучения.

---
#### Блок 1: Теория рекомендательных систем

Можно выделить два основных типа рекомендательных систем.

1. На основе контента (Content-based/Item-based)
	- Пользователю рекомендуются объекты, похожие на те, которые этот пользователь уже употребил.
	- Похожести оцениваются по признакам содержимого объектов.
	- Сильная зависимость от предметной области, полезность рекомендаций ограничена.

2. Коллаборативная фильтрация (Collaborative Filtering)
	- Для рекомендации используется история оценок как самого пользователя, так и других пользователей.
	- Более универсальный подход, часто дает лучший результат.
	- Есть свои проблемы (например, холодный старт).

В данной лабораторной работе мы будет рассматривать рекомендательную систему на основе контента 
в интернет магазине. В упрощённом варианте будем считать, что для такой рекомендательной системе 
основным источником информации для рекомендаций служат только эмбеддинги товаров. 

Чтобы понять, что такое ембеддинги (от англ. embedding), мы должны сначала понимать, что модели
машинного обучения могут принимать в качестве входных данных только числовые данные.

Это означает, что в такой области, как рекомендательные системы, мы должны преобразовывать нечисловые переменные,
такие как описания товаров, в числа и векторы, которые каким-то образом характеризует интересы контент. 
Такие векторы называются ембеддингами. 
""")

	st.image(
		image="data/web_images/embedding_example.png",
		caption="Например, эмбеддинги могут быть такими."
	)

	st.write("Ембеддинг обладает свойствами, как показано на картинке ниже."
		)

	st.image(
		image="data/web_images/embedding_properties.png",
		caption="Свойства эмбеддингов"
	)

	st.markdown(
		"""
        Хороший эмбеддинг должен хорошо описывать свойства объекта в числовом формате. Насколько хорошо ембеддинги
        описывают свойства объектов можно убедиться с помощью визуализации ембеддингов.
        """
	)

	st.image(
		image="data/web_images/good_embedding_visualisation.png",
		caption="Визуализация ембеддингов, которые хорошо описывают свойства объектов"
	)

	st.markdown(
		"""
        Не очень хорошие эмбеддинги могут указывать на похожесть слов или объектов, которые на самом деле
        семантически никак не связаны, или наоборот. При визуализации такие ембеддинги выглядят примерно
        следующим образом.
        """
	)

	st.image(
		image="data/web_images/not_good_embedding_visualisation.png",
		caption="Ембеддинги, которые не очень хорошо описывают свойства объектов"
	)
	st.write(
		"""
Оказывается, что на восприятие рекомендаций влияет не только качество ранжирования, 
но и некоторые другие характеристики. Среди них, например, разнообразие 
(не стоит выдавать пользователю фильмы только на одну тему или из одной серии), 
неожиданность (если рекомендовать очень популярные фильмы, то такие рекомендации 
будут слишком банальными и почти бесполезными), новизна (многим нравятся классические 
фильмы, но рекомендациями обычно пользуются, чтобы открыть для себя что-то новое) и многие другие.

---

#### Блок 2: Пример рекомендательной системы

##### Пример работы рекомендательной системы, основанной на эмбеддингах:
"""
	)


def show_rec_example():
	all_products = get_products_data()
	user_select = st.selectbox(
		label="Какую рекомендацию получить?",
		options=[
			"Для случайного продукта",
			"Выбрать из категории"

		]
	)

	if user_select == "Для случайного продукта":
		get_prediction_for_random_product = st.button("Получить рекомендации для случайного продукта")
		if get_prediction_for_random_product:
			random_product = get_random_product()
			product_index = random_product.name
			image = get_image_by_sku(product_index)
			st.image(image)
			st.write(f"Название: {random_product['dimension17']} {random_product['name']}")
			st.write(f"Цена: {random_product['price']} RUB")
			st.write(f"Описание: {random_product['description']}")

			# блок рекомендаций по описанию
			st.subheader("Список рекомендованного")
			indexes = requests.post(
				url=f"http://{API_HOST}:{API_PORT}/get_recommendation",
				params={"product_index": product_index}
			).json()["indexes"]

			# создаем 4 колонки по 2 товара из рекомендаций
			# в первую колонку попадают ближайшие эмбеддинги с нечетными индексами
			# во вторую колонку попадают ближайшие эмбеддинги с четными индексами
			for col, index_1 in zip(st.columns(4), range(1, 5)):
				# первая колонка с рекомендацией
				same = all_products.loc[indexes[index_1]]
				image = get_image_by_sku(same.name)
				col.image(
					image,
					caption=f"{same['dimension17'] if not isinstance(same['dimension17'], float) else ''} {same['name']} - {same['price']} RUB"
				)
				col.write(f"""
					##### {same['dimension17'].title() if not isinstance(same['dimension17'], float) else ''} {same['name']}
					{same['description'].capitalize()}
					"""[:200] + "..."
				)
				col.markdown("---")
				# вторая колонка с рекомендацией

			for col, index_2 in zip(st.columns(4), range(5, 9)):
				same = all_products.loc[indexes[index_2]]
				image = get_image_by_sku(same.name)
				col.image(
					image,
					caption=f"{same['dimension17'] if not isinstance(same['dimension17'], float) else ''} {same['name']} - {same['price']} RUB"
				)
				col.write(f"""
					##### {same['dimension17'].title() if not isinstance(same['dimension17'], float) else ''} {same['name']}
					{same['description'].capitalize()}
					"""[:200] + "..."
				)

	if user_select == "Выбрать из категории":
		# получаем список категорий с русским названием
		categories: list = st.multiselect("Выберете категорию продукта", options=get_category_options())
		# конвертируем список категорий в вид, в котором они содержатся в датафрейме
		eng_categories = [get_category_data("ru_to_eng", cat) for cat in categories]
		# если выбрана хотя бы одна категория
		if len(categories) > 0:
			# получаем все бренды из данной категории
			brands = st.multiselect(
				"Выберете бренд",
				options=list(all_products[all_products["category"].isin(eng_categories)]["brand"].value_counts().index)
			)
			# если выбран хотя бы один бренд
			if len(brands) > 0:
				# создаем кнопку выбора нового продукта
				change_product = st.button("Сменить продукт")
				# отфильтровываем продукты по выбранным пользователем критериям
				selected_products = all_products[
					all_products["category"].isin(eng_categories) & all_products["brand"].isin(brands)]
				# получаем рандомный продукт
				double_index = all_products[all_products["description"] == all_products["product_usage"]].index
				product = selected_products.loc[
					np.random.choice(list(set(selected_products.index).difference(set(double_index))))]
				# если юзер жмакнул кнопку сменить продукт
				if change_product:
					# выбираем новый продукт
					product = selected_products.loc[np.random.choice(selected_products.index)]
				image = get_image_by_sku(product.name)
				st.image(image)
				st.write(f"Название: {product['dimension17']} {product['name']}")
				st.write(f"Цена: {product['price']} RUB")
				st.write(f"Описание: {product['description']}")

				# блок рекомендаций по описанию
				st.subheader("Рекомендации по описанию продукта")
				indexes = requests.post(
					url=f"http://{API_HOST}:{API_PORT}/get_recommendation",
					params={"product_index": product.name}).json()["indexes"]

				# создаем 4 колонки по 2 товара из рекомендаций
				# в первую колонку попадают ближайшие эмбеддинги с нечетными индексами
				# во вторую колонку попадают ближайшие эмбеддинги с четными индексами
				for col, index_1 in zip(st.columns(4), range(1, 5)):
					# первая колонка с рекомендацией
					same = all_products.loc[indexes[index_1]]
					image = get_image_by_sku(same.name)
					col.image(
						image,
						caption=f"{same['dimension17'] if not isinstance(same['dimension17'], float) else ''} {same['name']} - {same['price']} RUB"
					)
					col.write(f"""
						##### {same['dimension17'].title() if not isinstance(same['dimension17'], float) else ''} {same['name']}
						{same['description'].capitalize()}
						"""[:200] + "..."
							  )
					col.markdown("---")
				# вторая колонка с рекомендацией

				for col, index_2 in zip(st.columns(4), range(5, 9)):
					same = all_products.loc[indexes[index_2]]
					image = get_image_by_sku(same.name)
					col.image(
						image,
						caption=f"{same['dimension17'] if not isinstance(same['dimension17'], float) else ''} {same['name']} - {same['price']} RUB"
					)
					col.write(f"""
						##### {same['dimension17'].title() if not isinstance(same['dimension17'], float) else ''} {same['name']}
						{same['description'].capitalize()}
						"""[:200] + "..."
							  )

def show_quiz():
	st.write(
		"""
        ---
        ТЕСТ
        """
	)

	with st.form(key="quiz"):
		total_answers = 0

		answer_1 = st.radio(
			label="Рекомендательные системы ...",
			options=[
				"Не приносят выгоду бизнесу и не помогают клиенту",
				"Приносят выгоду бизнесу и не помогают клиенту",
				"Приносят выгоду бизнесу и помогают клиенту",
				"Не приносят выгоду бизнесу и помогают клиенту"
			]
		)

		if answer_1 == "Приносят выгоду бизнесу и помогают клиенту":
			total_answers += 1

		answer_2 = st.radio(
			label="Что такое эмбеддинг?",
			options=[
				"Числовая форма представления нечисловых данных",
				"Нечисловая форма представления числовых данных"
			]
		)

		if answer_2 == "Числовая форма представления нечисловых данных":
			total_answers += 1

		answer_3 = st.radio(
			label="Что такое хороший эмбеддинг?",
			options=[
				"Чем больше значений, тем лучше эмбеддинг",
				"Чем меньше значений, тем лучше эмбеддинг",
				"Который описывает свойства объекта",
				"Нет такого понятия"
			]
		)

		if answer_3 == "Который описывает свойства объекта":
			total_answers += 1

		answer_4 = st.radio(
			label=("Какие выделяют два основных типа рекомендательных систем?"),
			options=[
				"На основе контента и Контентная фильтрация",
				"Пользовательская и контентная фильтрация",
				"Коллаборативная фильтрация и рекомендации на основе контента",
				"Рекомендации друзей и на основе контента"
			]
		)

		if answer_4 == "Коллаборативная фильтрация и рекомендации на основе контента":
			total_answers += 1

		answer_5 = st.radio(
			label="На основе каких оценок работает коллаборативная фильтрация",
			options=[
				"Оценок тайных покупателей и пользователей",
				"На основе уценок",
				"Оценки пользователей",
				"Оценки независимых экспертов",
				"Оценки коллабораторов"
			]
		)

		if answer_5 == "Оценки пользователей":
			total_answers += 1

		answer_6 = st.radio(
			label="Возможно ли визуализировать эмбеддинги?",
			options=[
				"Да",
				"Нет"
			]
		)

		if answer_6 == "Да":
			total_answers += 1

		answer_7 = st.radio(
			label="Стоит ли пользователю рекомендовать только максимально похожие товары?",
			options=[
				"Да",
				"Нет"
			]
		)

		if answer_7 == "Нет":
			total_answers += 1

		check = st.form_submit_button("Проверить тест")

	if check:
		if total_answers < 5:
			st.warning(
				"Следует еще подучить теорию."
			)
		else:
			st.success(
				"Отлично! Вы хорошо усвоили теоретический блок."
			)


def get_category_data(type: str, cat_name: str = None) -> int | list:
	"""
	Returns the requested data by category

	:param: type: Type returned data. Should be one of: id, ru_name or cat_names
	:param: cat_name: category name
	:return: returns the requested data
	"""

	cat = {
		"makijazh":              {"id": 3, "name": "макияж"},
		"uhod":                  {"id": 4, "name": "уход"},
		"volosy":                {"id": 6, "name": "волосы"},
		"parfjumerija":          {"id": 7, "name": "парфюмерия"},
		"zdorov-e-i-apteka":     {"id": 3747, "name": "здоровье и аптека"},
		"sexual-wellness":       {"id": 5962, "name": "sexual wellness"},
		"azija":                 {"id": 10, "name": "азия"},
		"organika":              {"id": 12, "name": "органика"},
		"dlja-muzhchin":         {"id": 3887, "name": "для мужчин"},
		"dlja-detej":            {"id": 4357, "name": "для детей"},
		"tehnika":               {"id": 3870, "name": "техника"},
		"dlja-doma":             {"id": 8202, "name": "для дома"},
		"odezhda-i-aksessuary":  {"id": 8529, "name": "одежда и аксессуары"},
		"nizhnee-bel-jo":        {"id": 8563, "name": "нижнее бельё"},
		"ukrashenija":           {"id": 5746, "name": "украшения"},
		"lajfstajl":             {"id": 8579, "name": "лайфстайл"},
		"ini-formaty":           {"id": 5159, "name": "тревел-форматы"},
		"tovary-dlja-zhivotnyh": {"id": 7638, "name": "товары для животных"}
	}

	ru_cat = {
		"макияж": "makijazh",
		"уход": "uhod",
		"волосы": "volosy",
		"парфюмерия": "parfjumerija",
		"здоровье и аптека": "zdorov-e-i-apteka",
		"sexual wellness": "sexual-wellness",
		"азия": "azija",
		"органика": "organika",
		"для мужчин": "dlja-muzhchin",
		"для детей": "dlja-detej",
		"техника": "tehnika",
		"для дома": "dlja-doma",
		"одежда и аксессуары": "odezhda-i-aksessuary",
		"нижнее бельё": "nizhnee-bel-jo",
		"украшения": "ukrashenija",
		"лайфстайл": "lajfstajl",
		"тревел-форматы": "ini-formaty",
		"товары для животных": "tovary-dlja-zhivotnyh"
	}

	if type not in ["cat_names", "id", "ru_name", "ru_to_eng"]:
		raise ValueError("Parameter type should by 'cat_names', 'id', 'ru_name' or 'ru_to_eng'")

	if type == "cat_names":
		return [cat for cat in cat.keys()]

	if cat_name is None:
		raise ValueError("Value cat_name should be not None")

	if type == "ru_to_eng":
		return ru_cat[cat_name]

	try:
		cat[cat_name]
	except KeyError:
		raise ValueError(f"Wrong category name: {cat_name}. \n"
						 f"Possible cat_names: {', '.join([name for name in cat.keys()])}")

	if type == "id":
		return cat[cat_name]["id"]
	elif type == "ru_name":
		return cat[cat_name]["name"]
	else:
		raise ValueError("type should be 'cat_names', 'id' or 'ru_name'")


@st.experimental_memo
def get_products_data() -> pd.DataFrame:
	data = pd.read_csv("data/products.csv", index_col="sku")
	return data.drop(data[data["category"] == "sexual-wellness"].index)


@st.experimental_memo
def get_image_data() -> pd.DataFrame:
	image_data = pd.read_csv("data/product_images.csv", index_col="sku")
	image_data = image_data.groupby(image_data.index)["image"].apply(list)
	return image_data


@st.experimental_memo
def get_category_options() -> list:
	"""
	Returns category list in RU lang

	:return: list of categories in RU lang
	"""
	data = get_products_data()
	ru_cat_names = [get_category_data("ru_name", cat) for cat in data["category"].value_counts().index]
	return ru_cat_names


def get_random_product() -> pd.Series:
	"""
	Returns random product

	:return: random product data
	"""
	data = get_products_data()
	res = data.loc[np.random.choice(data.index)]

	return res


def get_image_by_sku(sku: str or int) -> np.array:
	"""
	Get image by product sku.

	:param: sku: product sku
	:return: image product if existed, else image with 'No image' text
	"""
	image_data = get_image_data()
	try:
		for num, dir in enumerate(listdir("data/images")):
			try:
				image_name = image_data.loc[str(sku)][0]
				with Image.open(f"data/images/{dir}/{image_name}") as img:
					res = np.array(img)
				return res
			except:
				if num == 2:
					raise Exception
	except Exception as err:
		with Image.open("data/service_images/" + "no_img.jpg") as img:
			res = np.array(img)
		return res